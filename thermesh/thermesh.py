## Temperature mesh (subclass of mesh) - this defines a TriMesh plus temperature and various functions
    

import numpy as np
import math
from .. import TreMesh
from scipy import sparse
import time

class ThermMesh(TreMesh):
    """
    Builds a TreMesh (2D) object and adds a temperature field and data structures / operators
    to propagate information across the surface (e.g. flow downhill) 
    """
    
    name="Generic_Height_TreMesh"

    def __init__(self, points_x, points_y, temperature, diffusivity, heat_sources, boundary_mask, verbose=False):
        """
        Initialise the Delaunay mesh (parent) and build height data structures
        """
        
        # initialise the mesh itself from the parent TreMesh class
        TreMesh.__init__(self, points_x, points_y, boundary_mask, verbose=verbose)

        
        # Add the temperature field 
        self.temperature = temperature
        self.diffusivity = diffusivity
        self.heat_sources = heat_sources


        size = len(points_x)
        for nl in self.neighbourhood_array:
            size += len(nl)

        # Initialise matrices
        self.row_array = np.zeros(size, dtype=int)
        self.col_array = np.zeros(size, dtype=int)
        self.val_array = np.zeros(size)
        self.nPos = 0
        self.mCO_vector = np.zeros(len(points_x))

        self.BCs = dict()

        return


    def create_BC(self, name, value, indices, is_flux):
        """ Create a new BC with a given set of indices """

        if type(value) is float or int:
            value = value*np.ones_like(indices, dtype=float)
        elif len(value) == len(indices):
            pass
        else:
            raise ValueError('Length of indices / values do not match')

        self.BCs[str(name)] = {'value': value, 'indices': indices, 'is_flux': bool(is_flux)}
        

        if self.verbose:
            if is_flux:
                print " - Neumann boundary condition '%s' created" % name
            else:
                print " - Dirichlet boundary condition '%s' created" % name

        return


    def boundary_conditions(self, topBC, bottomBC, leftBC=0.0, rightBC=0.0):
        """
        1) Seperate boundary mask into quadrants
        2) Assign BC values (leftBC and rightBC are insulated by default)
        """

        inverse_bmask = np.invert(self.bmask)

        leftBC_coords = np.where(self.x == min(self.x[inverse_bmask]))[0]
        rightBC_coords = np.where(self.x == max(self.x[inverse_bmask]))[0]
        topBC_coords = np.where(self.y == max(self.y[inverse_bmask]))[0]
        bottomBC_coords = np.where(self.y == min(self.y[inverse_bmask]))[0]

        # Remove duplicates
        leftBC_coords = np.delete(leftBC_coords, [i for i,item in enumerate(leftBC_coords) if item in bottomBC_coords])
        leftBC_coords = np.delete(leftBC_coords, [i for i,item in enumerate(leftBC_coords) if item in topBC_coords])
        rightBC_coords = np.delete(rightBC_coords, [i for i,item in enumerate(rightBC_coords) if item in bottomBC_coords])
        rightBC_coords = np.delete(rightBC_coords, [i for i,item in enumerate(rightBC_coords) if item in topBC_coords])

        flux = np.zeros(4, dtype=bool)
        # Flux conditions in W/m^2 -- should be < 1
        for i, bc in enumerate([leftBC, rightBC, topBC, bottomBC]):
            if abs(bc) < 1:
                flux[i] = True

        self.create_BC('leftBC', leftBC, leftBC_coords, is_flux=flux[0])
        self.create_BC('rightBC', rightBC, rightBC_coords, is_flux=flux[1])
        self.create_BC('topBC', topBC, topBC_coords, is_flux=flux[2])
        self.create_BC('bottomBC', bottomBC, bottomBC_coords, is_flux=flux[3])

        return


    def initial_conditions(self):
        """
        Create an initial temperature field using an average gradient
        - This is useful for temperature-dependent solves
        """
        temperature = np.zeros_like(self.temperature)

        for row in range(len(self.x)):
            temperature[row,:] = self.topBC + gradient*dy

        self.temperature = temperature

        return temperature


    def delaunay_diffusion_rate(self, kappa, PHI):
        """
        Compute diffusive rate of change for field 'PHI' on the delaunay mesh
        
        """

        inverse_bmask = np.invert(self.bmask)
        
        diff_timestep   =  (self.area / kappa).min()

        ## Should wrap this as grad ( A * grad )

        gradZx, gradZy = self.delaunay_grad(PHI)
        
        flux_x = kappa * gradZx
        flux_y = kappa * gradZy    
        
        if self.fluxBC:
            flux_x[inverse_bmask] = 0.0
            flux_y[inverse_bmask] = 0.0  # outward normal flux, actually 
            
        diffDz  = self.delaunay_div(flux_x, flux_y)

        if not self.fluxBC:
            diffDz[inverse_bmask] = 0.0

        return diffDz, diff_timestep


    def implicit_conduction(self):
        """
        Solve steady-state conduction with a heat source term
        """
        t = time.clock()

        self.nPos = 0
        index = np.arange(len(self.x), dtype=int)

        ## Boundary conditions
        for key, attributes in self.BCs.iteritems():
            idx = 0
            if attributes['is_flux'] is True:
                for row in index[attributes['indices']]:
                    neighbours = self.neighbourhood_array[row]
                    cneighbours = self.closed_neighbourhood_array[row]
                    npoints = self.tri.points[neighbours]
                    cnpoints = self.tri.points[cneighbours]
                    nkappa = self.diffusivity[neighbours]
                    
                    delta, ad = np.empty(len(neighbours)), np.empty(len(neighbours))
                    for col, column in enumerate(neighbours):
                        coord = ( npoints[col-1] - cnpoints[col+1] )
                        delta[col] = math.hypot(coord[0], coord[1])
                        kappa = ( nkappa[col] + self.diffusivity[row] )
                        ad[col] = kappa / (2*delta[col]**2)
                        
                        self._writeMatrix(row, column, 1.0*ad[col])
                        
                    self._writeMatrix(row, row, np.sum(ad*-1))
                    self.mCO_vector[row] = attributes['value'][idx]/np.sum(delta) - self.heat_sources[row]
                    idx += 1

            else:
                for row in index[attributes['indices']]:
                    self._writeMatrix(row, row, 1.0)
                    self.mCO_vector[row] = attributes['value'][idx]
                    idx += 1



        ## Main body
        for row in index[self.bmask]:
            neighbours = self.neighbourhood_array[row]
            cneighbours = self.closed_neighbourhood_array[row]
            npoints = self.tri.points[neighbours]
            cnpoints = self.tri.points[cneighbours]
            nkappa = self.diffusivity[neighbours]
            
            ad = np.empty(len(neighbours))
            for col, column in enumerate(neighbours):
                coord = ( npoints[col-1] - cnpoints[col+1] )
                delta = math.hypot(coord[0], coord[1])
                kappa = ( nkappa[col] + self.diffusivity[row] )
                ad[col] = kappa / (2*delta**2)
                
                self._writeMatrix(row, column, 1.0*ad[col])
                
            self._writeMatrix(row, row, np.sum(ad*-1))
            self.mCO_vector[row] = -self.heat_sources[row]


        A = sparse.coo_matrix( (self.val_array, (self.row_array, self.col_array)) ).tocsr()
        b = sparse.coo_matrix( (np.array(self.mCO_vector) ), shape=(len(self.x), 1) ).T

        temperature = sparse.linalg.spsolve(A, b.tocsr())
        self.temperature = temperature


        if self.verbose:
            print " - Solving implicit conduction", (time.clock() - t), "s"

        return temperature


    def _writeMatrix(self, Ir, Jr, Vr):
        """ Writes nodes to a sparse tridiagonal matrix """
        self.row_array[self.nPos] = Ir
        self.col_array[self.nPos] = Jr
        self.val_array[self.nPos] = Vr
        self.nPos += 1
        return
