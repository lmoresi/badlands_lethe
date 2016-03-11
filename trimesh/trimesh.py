
##
## Python surface process modelling classes
##

from ..virtualmesh import VirtualMesh as VirtualMesh

import numpy as np
import math
import time

from scipy import sparse as sparse
from scipy.sparse import linalg as linalgs


class TriMesh(VirtualMesh):
    """
    Takes a cloud of points and finds the Delaunay mesh using qhull. Note that this is
    not a meshing algorithm as such since this only produces a mesh useful for computation
    if the boundary is convex.

    """

    def __init__(self):
        """
        Initialise the class
        """

        super(TriMesh, self).__init__()
        self.mesh_type="ConvexTriMesh"


    def build_mesh(self, points_x=None, points_y=None, boundary_mask=None, filename=None, **kwargs):
        """
        Initialise the triangulation and extend its data structures to include neighbour lists etc
        Enter keywords to pass to the triangulation operation.
        ** would be good to have the boundary mask specify the shape of the exterior.
        """

        from ..tools import Triangulation as __Triangulation
        import time

        if filename:
            self.read_from_file(filename)
        else:
            self.x = np.array(points_x)
            self.y = np.array(points_y)
            self.bmask = np.array(boundary_mask)

        walltime = time.clock()
        self.tri = __Triangulation(self.x, self.y)
        self.tri.triangulate(kwargs)
        if self.verbose:
            print " - Calculating Delaunay Triangulation ", time.clock() - walltime,"s"


        ## Construct the neighbour list which is absent from the Voronoi data structure

        walltime = time.clock()
        self.build_neighbours()
        if self.verbose:
            print " - Triangulation Neighbour Lists ", time.clock() - walltime,"s"

        ## Summation weights and local areas

        walltime = time.clock()
        self.build_node_weights_and_measures()
        if self.verbose:
            print " - Triangulation Local Areas and Weights ", time.clock() - walltime,"s"

        ## Matrix of gradient coefficients

        walltime = time.clock()
        self._delaunay_gradient_matrix()
        if self.verbose:
            print " - Triangulation Vector Operators ", time.clock() - walltime,"s"

        walltime = time.clock()
        self._matrix_build_local_area_smoothing_matrix()
        if self.verbose:
            print " - Local Smoothing Operator ", time.clock() - walltime,"s"

        return



    def read_from_file(self, filename, **kwargs):
        """
        Read TriMesh data from a file.
        Restores x, y, and triangulation information sufficient to rebuild the mesh.
        """

        if filename:
            try:
                meshdata = np.load(filename)

            except:
                print "Invalid mesh file - {s}".format(filename)

            else:
                meshdata = np.load(filename)
                self.x = meshdata['x']
                self.y = meshdata['y']
                self.bmask = meshdata['bmask']


    def write_to_file(self, filename, **kwargs):
        '''
        Save TriMesh data to a file - stores x, y and triangulation information sufficient to
        retrieve, plot and rebuild the mesh. Saves any given data

        '''

        np.savez(filename, x=self.x, y=self.y, bmask=self.bmask, triang=self.tri.simplices, **kwargs )


    def node_neighbours(self, centre_point):
        """
        Returns a list of neighbour nodes for a given point in the delaunay triangulation
        """

        return self.tri.vertex_neighbor_vertices[1][self.tri.vertex_neighbor_vertices[0][centre_point]:self.tri.vertex_neighbor_vertices[0][centre_point+1]]


    def build_neighbours(self):
        """
        1) Create a list of neighbour information (absent from the original tri data structures)
        2) Create an np.array with information needed to create matrices of interaction coefficients
           for computation (i.e. include the central node as well as the neighbours) - this is important
           when computing derivatives at boundaries for example.
        """
        from scipy.spatial import cKDTree
        import time

        # walltime = time.clock()
        bins = np.bincount(self.tri.simplices.flatten())
        placeholder = np.zeros(self.tri.simplices.size, dtype=bool)
        tree = cKDTree(np.column_stack([placeholder,self.tri.simplices.flatten()]))

        d, index = tree.query(np.column_stack([np.zeros(self.tri.npoints, dtype=bool),
                                               np.arange(self.tri.npoints)]), k=bins.max())

        neighbour_list = []

        for node in xrange(self.tri.npoints):
            simplices_subset = self.tri.simplices[index[node,:bins[node]]//3]
            neighbours = np.unique(simplices_subset)
            neighbour_list.append(neighbours[neighbours != node])

        self.neighbour_list = neighbour_list
        neighbour_array = np.array(self.neighbour_list)

        # And now a closed polygon of the encircling neighbours (include self if on boundary)
        # To use this for integration etc, we need an ordered list

        closed_neighbourhood_array = np.array(self.neighbour_list)
        neighbourhood_array = np.array(self.neighbour_list)

        # print "  Unsorted neighbours - ", time.clock() - walltime,"s"
        # walltime = time.clock()

        for node, node_array in enumerate(closed_neighbourhood_array):
            neighbour_array[node] = np.hstack( (node, node_array) )

            # Boundary nodes, the encircling nodes includes the node itself
            if not self.bmask[node]:
                node_array = neighbour_array[node]

            # Now order the list (use centroid since the node is included in boundary loops)

                xx = self.x[node_array]
                yy = self.y[node_array]
                cx = xx.mean() #!!
                cy = yy.mean() #!!
                rx = xx - cx
                ry = yy - cy

                ordering = np.arctan2(rx, ry).argsort()

            else:
                xx = self.x[node_array] - self.x[node]
                yy = self.y[node_array] - self.y[node]

                ordering = np.arctan2(xx, yy).argsort()

            neighbourhood_array[node] = node_array[ordering]

            # Now close the polygon

            closed_neighbourhood_array[node] = np.hstack( (neighbourhood_array[node], neighbourhood_array[node][0]) )

        self.neighbour_array = neighbour_array
        # print "  Closed, sorted neighbours - ", time.clock() - walltime,"s"


        self.closed_neighbourhood_array = closed_neighbourhood_array
        self.neighbourhood_array = neighbourhood_array


        return

    def build_node_weights_and_measures(self):
        """
        Stores the local areas and the local weights for summation for each
        point in the Delaunay triangulation
        """

        ntriw = np.zeros(self.tri.npoints)

        for idx, triangle in enumerate(self.tri.simplices):
            coords = self.tri.points[triangle]
            vector1 = coords[1] - coords[0]
            vector2 = coords[2] - coords[0]
            ntriw[triangle] += abs(vector1[0]*vector2[1] - vector1[1]*vector2[0])

        self.area = ntriw / 6.0
        self.weight = 1.0 / ntriw

        return

    def _barycentric_coords(self, triangle, coord):
        """
        For a given triangle (2D simplex) in the triangulation, return the
        barycentric coordinates for the coordinate.

        This function is probably only useful for precomputation.
        """

        local = coord - self.tri.transform[triangle,2,:]
        bary = self.tri.transform[triangle,:2,:2].dot(local)
        return np.array([bary[0], bary[1], 1-bary.sum()])

    def _interpolate_bc(self, data, triangle, bary_coord):
        return bary_coord.dot( data[self.tri.simplices[triangle]] )

    def interpolate(self, data, coord, error_value=0.0):
        """
        Interpolates the data array from the points of the triangulation to an arbitrary coord
        within the domain. If the coord is not within the domain then error_value is returned along with
        a False flag.

        e.g. value, success = trimesh.interpolate( data_array, (x,y) , 0.0 )
        """

        ## Should check that data is a suitable sized array

        triangle = self.tri.find_simplex(coord)

        if triangle == -1:
            return error_value, False

        ## See the documentation for scipy.spatial for an explanation of this:

        local = coord - self.tri.transform[triangle,2,:]
        bary = self.tri.transform[triangle,:2,:2].dot(local)
        bc = np.array([bary[0], bary[1], 1-bary.sum()])

        values = data[self.tri.simplices[triangle]]
        interpolated_value = values.dot(bc)

        return interpolated_value, True

##  Differential operators on the triangulation



    def derivative_grad(self, PHI):

        return self.gradMx.dot(PHI) , self.gradMy.dot(PHI)


    def derivative_div(self, PSIx, PSIy):
        """
        Constructs the divergence (div ) of a Vector (PSIx, PSIy) using equivalent to
        self._matrix_delaunay_grad(). Optimised routine using sparse matrix gradient operator
        """
        return self.gradMx.dot(PSIx) + self.gradMy.dot(PSIy)

    def derivative_del2(self, PHI):
        """
        Constructs the laplacian (div grad) of of a scalar (PHI) using equivalent to
        self.delaunay_grad() and self.delaunay_div(). Optimised routine using
        sparse matrix gradient operator
        """

        return self.gradM2.dot(PHI)


    def _delaunay_gradient_matrix(self):
        """
        Creates the sparse matrix form of the gradient operator (and del squared) from the unstructured grid.
            self.gradMx,
            self.gradMy,
            self.gradM2
        This routine does not construct the array equivalents (see _array_store_delaunay_grad_matrix )
        """

        t = time.clock()

        size = 0
        for nl in self.neighbourhood_array:
            size += len(nl)

        row_array = np.empty(size, dtype = int)
        col_array = np.empty(size, dtype = int)
        grad_x_array = np.zeros(size)
        grad_y_array = np.zeros(size)

        idx=0
        for row in range(0, len(self.neighbourhood_array)):
            neighbours  = self.neighbourhood_array[row]
            cneighbours = self.closed_neighbourhood_array[row]
            npoints  = self.tri.points[neighbours]
            cnpoints = self.tri.points[cneighbours]

            for col, column in enumerate(neighbours):

                coord1m  =  npoints[col-1]
                coord1p  = cnpoints[col+1]

                delta =   (coord1p - coord1m)

                row_array[idx] = row
                col_array[idx] = column
                grad_x_array[idx] =  delta[1] * self.weight[column]
                grad_y_array[idx] = -delta[0] * self.weight[column]

                idx += 1

        # We can re-pack this array into a sparse matrix for v. fast computation of gradient operators


        gradMxCOO  = sparse.coo_matrix( (grad_x_array, (row_array, col_array)) ).T
        gradMyCOO  = sparse.coo_matrix( (grad_y_array, (row_array, col_array)) ).T

        gradMx = gradMxCOO.tocsr()
        gradMy = gradMyCOO.tocsr()
        gradM2 = gradMx.dot(gradMx) + gradMy.dot(gradMy) # The del^2 operator !

        self.gradMx = gradMx
        self.gradMy = gradMy
        self.gradM2 = gradM2

        return


## FIX !!


    def _matrix_build_local_area_smoothing_matrix(self):
        """

        """

        t = time.clock()

        size = 0
        for nl in self.neighbour_array:
            size += len(nl)

        row_array = np.empty(size, dtype = int)
        col_array = np.empty(size, dtype = int)
        smooth_array = np.zeros(size)


        idx=0
        for row in range(0, len(self.neighbour_array)):
            neighbours  = self.neighbour_array[row]
            weight =  1.0/len(neighbours)

            for col, column in enumerate(neighbours):
                row_array[idx] = row
                col_array[idx] = column

                smooth_array[idx] = weight

                # if row==column:
                #     smooth_array[idx] = 0.5
                # else:
                #     smooth_array[idx] =  weight

                idx += 1

        # We can re-pack this array into a sparse matrix for v. fast computation of gradient operators


        smoothCOO  = sparse.coo_matrix( (smooth_array, (row_array, col_array)) )

        smoothMat = smoothCOO.tocsr()

        self.localSmoothMat = smoothMat


        return


    def local_area_smoothing(self, data, its=1, centre_weight=0.75):

        smooth_data = data.copy()

        for i in range(0, its):
            smooth_data = centre_weight * smooth_data + (1.0-centre_weight) * self.localSmoothMat.dot(smooth_data)

        return smooth_data



    ## We can build a matrix version of this or some similar smoothing kernel

    def local_area_smoothing2(self, data, its):

        smooth_data = data.copy()
        mean_data = np.zeros_like(data)

        for it in range(0, its):
            for i in range(0, self.tri.npoints):
                mean_data[i] =  smooth_data[self.neighbour_array[i]].mean()

            smooth_data = mean_data.copy()

        return smooth_data


    def add_node_data_to_plot(self, this_plot_ax, data, shading="gouraud", **kwargs ):
        """
        Adds a tripcolor plot of node data on this mesh to an existing plot (matplotlib axis)
        No error checking - that is handled by matplotlib.
        """

        sm1 = this_plot_ax.tripcolor(self.x, self.y, self.tri.simplices.copy(), data, shading=shading, **kwargs)

        return sm1

    def add_node_contours_to_plot(self, this_plot_ax, data, *args, **kwargs ):
        """
        Adds a tripcolor plot of node data on this mesh to an existing plot (matplotlib axis)
        No error checking - that is handled by matplotlib.
        """

        sm1 = this_plot_ax.tricontour(self.x, self.y, self.tri.simplices.copy(), data, *args, **kwargs)

        return sm1


    def assess_derivative_quality(self):
        """
        Load a gaussian function onto the mesh and test the derivatives
        Error = norm2(Numerical - Analytic) / norm2(Analytic)
        """

        import numpy.linalg as npl

        ## Would be useful to supply a function of (x,y) and relevant derivatives and have this routine run the test.

        # Z = np.empty(self.tri.npoints)
        Z = np.exp(-self.x**2 -self.y**2)
        gradZx = -2 * self.x * Z
        gradZy = -2 * self.y * Z
        del2Z  =  4 * Z * ( self.x**2 + self.y**2 - 1.0)

        NgradZx,NgradZy = self.derivative_grad(Z)
        Ndel2Z = self.derivative_div(NgradZx,NgradZy)

        gradError = (npl.norm(gradZx-NgradZx,2) + npl.norm(gradZy-NgradZy,2) ) / (npl.norm(gradZx,2) + npl.norm(gradZy,2))
        del2Error = npl.norm(del2Z-Ndel2Z,2) / npl.norm(del2Z,2)

        return gradError, del2Error
