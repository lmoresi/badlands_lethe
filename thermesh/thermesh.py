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
            
        return


    def boundary_conditions(self, topBC, bottomBC, leftBC=0.0, rightBC=0.0, fluxBC=False):
        """
        Seperate boundary mask into quadrants to assign BCs
        DO NOT use Lloyd's mesh improvement algorithm!
        """

        self.leftBC = leftBC
        self.rightBC = rightBC
        self.topBC = topBC
        self.bottomBC = bottomBC

        self.fluxBC = fluxBC

        leftBC_mask = np.zeros_like(self.bmask, dtype=bool)
        rightBC_mask = np.zeros_like(self.bmask, dtype=bool)
        topBC_mask = np.zeros_like(self.bmask, dtype=bool)
        bottomBC_mask = np.zeros_like(self.bmask, dtype=bool)

        leftBC_mask[np.where(self.x == min(self.x))] = True
        rightBC_mask[np.where(self.x == max(self.x))] = True
        topBC_mask[np.where(self.y == max(self.y))] = True
        bottomBC_mask[np.where(self.y == min(self.y))] = True

        # check to make sure we haven't got too many elements
        # if np.sum(self.bmask) != np.sum(topBC_mask)+np.sum(bottomBC_mask)+np.sum(leftBC_mask)+np.sum(rightBC_mask):
            # print "boundary mask is irregular"

        self.leftBC_mask = leftBC_mask
        self.rightBC_mask = rightBC_mask
        self.topBC_mask = topBC_mask
        self.bottomBC_mask = bottomBC_mask

        return

    def initial_conditions(self):
        """
        Create an initial temperature field using an average gradient
        """
        temperature = np.zeros_like(self.temperature)

        for row in range(len(self.neighbourhood_array)):
            temperature[row,:] = self.topBC + gradient*dy

        self.temperature = temperature

        return

    def temperature_dependent(self):
        """
        Apply an operation to the conductivity field using a user-defined function
        """
        self.temperature_dependent = True

        # conductivity = self.conductivity * (298./self.temperature)**

        return


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

        def writeMatrix(Ir, Jr, Vr):
            """ Writes nodes to a sparse tridiagonal matrix """
            global row_array, col_array, val_array, nPos
            row_array[nPos] = Ir
            col_array[nPos] = Jr
            val_array[nPos] = Vr
            nPos += 1

        size = 0
        for nl in self.neighbourhood_array:
            size += len(nl)

        global row_array, col_array, val_array, nPos
        row_array = np.empty(size*5, dtype=int)
        col_array = np.empty(size*5, dtype=int)
        val_array = np.zeros(size*5)
        nPos = 0
        mCO_vector = np.zeros(size*5)

        idx=0    
        for row in range(0, len(self.neighbourhood_array)):    
            neighbours  = self.neighbourhood_array[row]  
            cneighbours = self.closed_neighbourhood_array[row] 
            npoints  = self.tri.points[neighbours]
            cnpoints = self.tri.points[cneighbours]

            for col, column in enumerate(neighbours):  

                Di = [self.diffusivity[row-1,column], self.diffusivity[row,column], self.diffusivity[row+1,column]]
                Dj = [self.diffusivity[row,column-1], self.diffusivity[row,column], self.diffusivity[row,column+1]]

                coord1m  =  npoints[col-1]
                coord1p  = cnpoints[col+1]

                delta =   (coord1p - coord1m)
                adx = 1.0 / (2*delta[1]**2)
                ady = 1.0 / (2*delta[0]**2)

                # row_array[idx] = row
                # col_array[idx] = column
                # val_array[idx] =  delta[1] * self.weight[column]
                # mCO_vector[idx] = -delta[0] * self.weight[column]

                writeMatrix(index[idx], index[idx]-nx, (Dj[0]+Dj[1])*ady)
                writeMatrix(index[idx], index[idx]-1, (Di[0]+Di[1])*adx)
                writeMatrix(index[idx], index[idx], (Di[0]+2*Di[1]+Di[2])*-adx + (Dj[0]+2*Dj[1]+Dj[2])*-ady)
                writeMatrix(index[idx], index[idx]+1, (Di[2]+Di[1])*adx)
                writeMatrix(index[idx], index[idx]+nx, (Dj[2]+Dj[1])*ady)

                matC[index[idx]] = -heatProduction[r,c]

                idx += 1


        A = sparse.coo_matrix( (val_array, (row_array, col_array)) ).tocsr()
        b = sparse.coo_matrix( (mCO_vector), shape=(row_array, 1) ).T

        temperature = sparse.linalg.spsolve(A, b.tocsr())
        # self.temperature = temperature

        return temperature