
## 
## Python surface process modelling classes
##

import numpy as np
import math
import time

from scipy import sparse as sparse
from scipy.sparse import linalg as linalgs     
 

class TreMesh:
    """
    Takes a cloud of points and finds the Delaunay mesh using qhull. Note that this is 
    not a meshing algorithm as such since this only produces a mesh useful for computation
    if the boundary is convex. 
    
    """

    
    name = "Generic_TreMesh"
        
    def __init__(self, points_x, points_y, boundary_mask, verbose=None):
        """ 
        Initialise the triangulation and extend its data structures to include neighbour lists etc
        """

        from scipy.spatial import Delaunay as __Delaunay
        import time

        
        self.x = np.array(points_x)
        self.y = np.array(points_y)
        self.bmask = np.array(boundary_mask)
        self.verbose = verbose
        
        # multiple possible implementations of the vector operators
        # Note the need to store the correct matrices / arrays
        
        self.delaunay_grad = self._matrix_delaunay_grad
        self.delaunay_div  = self._matrix_delaunay_div
        self.delaunay_del2 = self._matrix_delaunay_del2
        self._store_delaunay_grad_matrix = self._matrix_store_delaunay_grad_matrix
        
        # self.delaunay_grad = self._array_delaunay_grad
        # self.delaunay_div  = self._array_delaunay_div
        # self.delaunay_del2 = self._array_delaunay_del2
        # self._store_delaunay_grad_matrix = self._array_store_delaunay_grad_matrix
        
        walltime = time.clock()
        points = np.column_stack((self.x, self.y))
        self.tri = __Delaunay(points)
        if self.verbose:
            print " - Calculating Delaunay Triangulation ", time.clock() - walltime,"s"
  
        
        ## Construct the neighbour list which is absent from the Voronoi data structure
         
        walltime = time.clock()            
        self._store_neighbour_information()          
        if self.verbose:
            print " - Triangulation Neighbour Lists ", time.clock() - walltime,"s"
         
        ## Summation weights and local areas    
        
        walltime = time.clock()            
        self._store_weights_and_measures()
        if self.verbose:
            print " - Triangulation Local Areas and Weights ", time.clock() - walltime,"s"
              
        ## Matrix of gradient coefficients    
        
        walltime = time.clock()            
        self._store_delaunay_grad_matrix()
        if self.verbose: 
            print " - Triangulation Vector Operators ", time.clock() - walltime,"s"
  
        walltime = time.clock()            
        self._matrix_build_local_area_smoothing_matrix()
        if self.verbose: 
            print " - Local Smoothing Operator ", time.clock() - walltime,"s"

        return
 
    
    def neighbours(self, centre_point):
        """
        Returns a list of neighbour nodes for a given point in the delaunay triangulation
        """
        
        return self.tri.vertex_neighbor_vertices[1][self.tri.vertex_neighbor_vertices[0][centre_point]:self.tri.vertex_neighbor_vertices[0][centre_point+1]]


    def _store_neighbour_information(self):
        """
        1) Create a list of neighbour information (absent from the original tri data structures)
        2) Create an np.array with information needed to create matrices of interaction coefficients
           for computation (i.e. include the central node as well as the neighbours) - this is important
           when computing derivatives at boundaries for example.
        """
        
        import time

        # walltime = time.clock()

        neighbour_list = []
        num_neighbours = np.zeros(len(self.tri.points), dtype=int)

        for node in range(0,len(self.tri.points)):
            neighbours = self.neighbours(node)
            num_neighbours[node] = len(neighbours)
            neighbour_list.append(neighbours)

        self.neighbour_list = neighbour_list
          
        neighbour_array = np.array(self.neighbour_list)

        for node, node_array in enumerate(neighbour_array):
            neighbour_array[node] = np.hstack( (node, node_array) )

        self.neighbour_array = neighbour_array
 
        # And now a closed polygon of the encircling neighbours (include self if on boundary)
        # To use this for integration etc, we need an ordered list
        
        closed_neighbourhood_array = np.array(self.neighbour_list)
        neighbourhood_array = np.array(self.neighbour_list)

        # print "  Unsorted neighbours - ", time.clock() - walltime,"s"
        # walltime = time.clock()
        
        for node, node_array in enumerate(closed_neighbourhood_array):
            # Boundary nodes, the encircling nodes includes the node itself
            if not self.bmask[node]:
                node_array = np.hstack( (node_array, node) )

            # Now order the list (use centroid since the node is included in boundary loops)
            locations =  np.array(self.tri.points[node_array].T)
            centroid =   np.array( (locations[0].mean() , locations[1].mean()) )
            rlocations = (locations.T - centroid).T
            theta    = np.arctan2(rlocations[0], rlocations[1])    
            ordering = np.argsort(theta)

            neighbourhood_array[node] = node_array[ordering]

            # Now close the polygon
            
            closed_neighbourhood_array[node] = np.hstack( (neighbourhood_array[node], neighbourhood_array[node][0]) )     
            
            
        # print "  Closed, sorted neighbours - ", time.clock() - walltime,"s"
    

        self.closed_neighbourhood_array = closed_neighbourhood_array
        self.neighbourhood_array = neighbourhood_array

      
        return
    
    def _store_weights_and_measures(self):                                   
        """
        Stores the local areas and the local weights for summation for each 
        point in the Delaunay triangulation
        """
            
        ntriw = np.zeros(self.tri.npoints)
        area  = np.zeros(self.tri.npoints)

        for idx, triangle in  enumerate(self.tri.simplices):
            coords = self.tri.points[ triangle ]
            vector1 = coords[1] - coords[0]
            vector2 = coords[2] - coords[0]
            ntriw[triangle] += abs(vector1[0]*vector2[1] - vector1[1]*vector2[0])
            # area[triangle]  += abs(vector1[0]*vector2[1] - vector1[1]*vector2[0]) / 6.0

        area = ntriw / 6.0 
        ntriw = 1.0 / ntriw   

        self.area = np.array(area)
        self.weight = np.array(ntriw)

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
    
    def _slow_delaunay_grad(self, PHI):
        """
        Constructs the gradient of a scalar (PHI) from a path integral around the 
        neighbour-nodes of the Delaunay triangulation. 
            
        """
    
        ntridzdx = np.zeros(len(self.tri.points))
        ntridzdy = np.zeros(len(self.tri.points))

        for idx, triangle in  enumerate(self.tri.simplices):
            coords = self.tri.points[ triangle ]
            
            ## Note that the weights cancel the area correctly 

            centroid_dx =  ( PHI[triangle[0]] * (coords[1][1] - coords[2][1]) + 
                             PHI[triangle[1]] * (coords[2][1] - coords[0][1]) + 
                             PHI[triangle[2]] * (coords[0][1] - coords[1][1]) ) # / abs(vector1[0]*vector2[1] - vector1[1]*vector2[0])


            centroid_dy =  ( PHI[triangle[0]] * (coords[2][0] - coords[1][0]) + 
                             PHI[triangle[1]] * (coords[0][0] - coords[2][0]) + 
                             PHI[triangle[2]] * (coords[1][0] - coords[0][0]) ) # / abs(vector1[0]*vector2[1] - vector1[1]*vector2[0])

            ntridzdx[triangle] += centroid_dx
            ntridzdy[triangle] += centroid_dy

        ntridzdx *= self.weight 
        ntridzdy *= self.weight

        return ntridzdx, ntridzdy
    
    def _array_delaunay_grad(self, PHI ):
        """
        Constructs the gradient of a scalar (PHI) from a path integral around the 
        neighbour-nodes of the Delaunay triangulation. Optimised routine using 
        stored gradient operator coefficients        
        """
        
        gradPHIxm = np.zeros(self.tri.npoints)
        gradPHIym = np.zeros(self.tri.npoints)

        for node in range(0,self.tri.npoints):
            gradPHIxm[self.neighbour_array[node]] +=  PHI[node] * self.grad_matrix_x[node] 
            gradPHIym[self.neighbour_array[node]] +=  PHI[node] * self.grad_matrix_y[node] 

        gradPHIxm *= self.weight
        gradPHIym *= self.weight  

        return gradPHIxm, gradPHIym
    
    
    def _matrix_delaunay_grad(self, PHI):
        
        return self.gradMx.dot(PHI) , self.gradMy.dot(PHI)


    # gradA.gradB at the centroid level

    def _slow_delaunay_gradAB(self, A, B):      
        """
        Constructs the dot product of the gradient of two scalars (i.e. grad(A) . grad(B) )
        using the approach in self.delaunay_grad()
        """

        ntridAB = np.zeros(len(self.tri.points))

        for idx, triangle in  enumerate(self.tri.simplices):
            coords = self.tri.points[ triangle ]

            centroid_Adx =  (A[triangle[0]] * (coords[1][1] - coords[2][1]) + 
                             A[triangle[1]] * (coords[2][1] - coords[0][1]) + 
                             A[triangle[2]] * (coords[0][1] - coords[1][1]) ) # / abs(vector1[0]*vector2[1] - vector1[1]*vector2[0])


            centroid_Ady =  (A[triangle[0]] * (coords[2][0] - coords[1][0]) + 
                             A[triangle[1]] * (coords[0][0] - coords[2][0]) + 
                             A[triangle[2]] * (coords[1][0] - coords[0][0]) ) # / abs(vector1[0]*vector2[1] - vector1[1]*vector2[0])

            centroid_Bdx =  (B[triangle[0]] * (coords[1][1] - coords[2][1]) + 
                             B[triangle[1]] * (coords[2][1] - coords[0][1]) + 
                             B[triangle[2]] * (coords[0][1] - coords[1][1]) ) # / abs(vector1[0]*vector2[1] - vector1[1]*vector2[0])


            centroid_Bdy =  (B[triangle[0]] * (coords[2][0] - coords[1][0]) + 
                             B[triangle[1]] * (coords[0][0] - coords[2][0]) + 
                             B[triangle[2]] * (coords[1][0] - coords[0][0]) ) # / abs(vector1[0]*vector2[1] - vector1[1]*vector2[0])



            ntridAB[triangle] += centroid_Adx * centroid_Bdx + centroid_Ady * centroid_Bdy 

        ntridAB *= self.weight

        return ntridAB
    
 
    def _slow_delaunay_div(self, PSIx, PSIy):
        """
        Constructs the divergence of a vector (PSI) from a path integral around the 
        neighbour-nodes of the Delaunay triangulation. 
            
        """

        ntridiv = np.zeros(self.tri.npoints)

        for idx, triangle in  enumerate(self.tri.simplices):
            coords = self.tri.points[ triangle ]

            centroid_dx =  ( PSIx[triangle[0]] * (coords[1][1] - coords[2][1]) + 
                             PSIx[triangle[1]] * (coords[2][1] - coords[0][1]) + 
                             PSIx[triangle[2]] * (coords[0][1] - coords[1][1]) ) 

            centroid_dy =  ( PSIy[triangle[0]] * (coords[2][0] - coords[1][0]) + 
                             PSIy[triangle[1]] * (coords[0][0] - coords[2][0]) + 
                             PSIy[triangle[2]] * (coords[1][0] - coords[0][0]) ) 

            ntridiv[triangle] += centroid_dx + centroid_dy

        ntridiv *= self.weight

        return ntridiv

    def _array_delaunay_div(self, PSIx, PSIy ):
        """
        Constructs the divergence of a vector (PSI) from a path integral around the 
        neighbour-nodes of the Delaunay triangulation. Optimised routine using 
        stored gradient operator coefficients                 
        """

        divPSI = np.zeros(self.tri.npoints)

        for node in range(0,self.tri.npoints):
            divPSI[self.neighbour_array[node]] +=  PSIx[node] * self.grad_matrix_x[node] + PSIy[node] * self.grad_matrix_y[node]

        divPSI *= self.weight

        return divPSI
    
    
    def _matrix_delaunay_div(self, PSIx, PSIy):
        """
        Constructs the divergence (div ) of a Vector (PSIx, PSIy) using equivalent to
        self._matrix_delaunay_grad(). Optimised routine using sparse matrix gradient operator           
        """
        return self.gradMx.dot(PSIx) + self.gradMy.dot(PSIy)

   
   
    # This one is div(VA) when N is a vector, A a scalar
    
    def _slow_delaunay_divVA(self, Vx, Vy, A):
        """
        Constructs the divergence of the product of a scalar (A) with a vector (V) i.e. 
        div( A V ) using the approach in self.delaunay_div()
        """

        ntridVA = np.zeros(self.tri.npoints)

        for idx, triangle in  enumerate(self.tri.simplices):
            coords = self.tri.points[ triangle ]

            centroid_AVx =  ( A[triangle[0]] * Vx[triangle[0]] * (coords[1][1] - coords[2][1]) + 
                              A[triangle[0]] * Vx[triangle[1]] * (coords[2][1] - coords[0][1]) + 
                              A[triangle[0]] * Vx[triangle[2]] * (coords[0][1] - coords[1][1]) ) # / abs(vector1[0]*vector2[1] - vector1[1]*vector2[0])


            centroid_AVy =  ( A[triangle[0]] * Vy[triangle[0]] * (coords[2][0] - coords[1][0]) + 
                              A[triangle[0]] * Vy[triangle[1]] * (coords[0][0] - coords[2][0]) + 
                              A[triangle[0]] * Vy[triangle[2]] * (coords[1][0] - coords[0][0]) ) # / abs(vector1[0]*vector2[1] - vector1[1]*vector2[0])



            ntridVA[triangle] += centroid_AVx + centroid_AVy 

        ntridVA *= self.weight

        return ntridVA
 
    def _slow_delaunay_del2(self, PHI):
        """
        Constructs the laplacian (div grad) of of a scalar (PHI) using
        self.delaunay_grad() and self.delaunay_div()
        """
        PHIx, PHIy = self._slow_delaunay_grad(PHI)
        return self._slow_delaunay_div(PHIx, PHIy)    
    
    
    def _array_delaunay_del2(self, PHI ):
        """
        Constructs the laplacian (div grad) of of a scalar (PHI) using equivalent to
        self.delaunay_grad() and self.delaunay_div(). Optimised routine using 
        stored gradient operator coefficients          
        """
        
        del2PHI   = np.zeros(self.tri.npoints)
        gradPHIxm = np.zeros(self.tri.npoints)
        gradPHIym = np.zeros(self.tri.npoints)

        # Works

        for node in range(0,self.tri.npoints):
            gradPHIxm[self.neighbour_array[node]] +=  PHI[node] * self.grad_matrix_x[node] 
            gradPHIym[self.neighbour_array[node]] +=  PHI[node] * self.grad_matrix_y[node] 

        gradPHIxm *= self.weight
        gradPHIym *= self.weight  

        for node in range(0,self.tri.npoints):
            del2PHI[self.neighbour_array[node]] +=  gradPHIxm[node] * self.grad_matrix_x[node] + gradPHIym[node] * self.grad_matrix_y[node]

        del2PHI *= self.weight

        return del2PHI    
    
    def _matrix_delaunay_del2(self, PHI):
        """
        Constructs the laplacian (div grad) of of a scalar (PHI) using equivalent to
        self.delaunay_grad() and self.delaunay_div(). Optimised routine using 
        sparse matrix gradient operator           
        """
             
        return self.gradM2.dot(PHI) 

   
  
    def _array_store_delaunay_grad_matrix(self):

        ntridzdx = np.zeros(self.tri.npoints)
        ntridzdy = np.zeros(self.tri.npoints)

        grad_matrix_x = np.array(self.neighbour_array) * 0.0   # same shape, convert to float
        grad_matrix_y = np.array(self.neighbour_array) * 0.0
        
        t=time.clock()

        for idx, triangle in  enumerate(self.tri.simplices):

            coords = self.tri.points[ triangle ]

            node0_dx = (coords[1][1] - coords[2][1])
            node1_dx = (coords[2][1] - coords[0][1])
            node2_dx = (coords[0][1] - coords[1][1])

            node0_dy = (coords[2][0] - coords[1][0])
            node1_dy = (coords[0][0] - coords[2][0])
            node2_dy = (coords[1][0] - coords[0][0]) 

            node0 = triangle[0]
            node1 = triangle[1]
            node2 = triangle[2]

            neighbours0 = self.neighbour_array[node0]
            neighbours1 = self.neighbour_array[node1]
            neighbours2 = self.neighbour_array[node2]

            grad_matrix_x[node0][0] += node0_dx  # Me !
            grad_matrix_x[node0][np.where(neighbours0 == node1)[0]] += node0_dx
            grad_matrix_x[node0][np.where(neighbours0 == node2)[0]] += node0_dx
            grad_matrix_y[node0][0] += node0_dy  # Me !
            grad_matrix_y[node0][np.where(neighbours0 == node1)[0]] += node0_dy
            grad_matrix_y[node0][np.where(neighbours0 == node2)[0]] += node0_dy

            grad_matrix_x[node1][0] += node1_dx  # Me !
            grad_matrix_x[node1][np.where(neighbours1 == node0)[0]] += node1_dx
            grad_matrix_x[node1][np.where(neighbours1 == node2)[0]] += node1_dx
            grad_matrix_y[node1][0] += node1_dy  # Me !
            grad_matrix_y[node1][np.where(neighbours1 == node0)[0]] += node1_dy
            grad_matrix_y[node1][np.where(neighbours1 == node2)[0]] += node1_dy

            grad_matrix_x[node2][0] += node2_dx  # Me !
            grad_matrix_x[node2][np.where(neighbours2 == node0)[0]] += node2_dx
            grad_matrix_x[node2][np.where(neighbours2 == node1)[0]] += node2_dx
            grad_matrix_y[node2][0] += node2_dy  # Me !
            grad_matrix_y[node2][np.where(neighbours2 == node0)[0]] += node2_dy
            grad_matrix_y[node2][np.where(neighbours2 == node1)[0]] += node2_dy
            
            
        if self.verbose:
            print "    Calculation of grad arrays ", time.clock()-t, "Seconds "

        # We can re-pack this array into a sparse matrix for v. fast computation of gradient operators    
        
        t = time.clock()
            
        size = 0
        for nl in self.neighbour_array:
            size += len(nl)

        row_array = np.empty(size, dtype = int)
        col_array = np.empty(size, dtype = int)
        grad_x_array = np.empty(size)
        grad_y_array = np.empty(size)

        idx=0    
        for row in range(0, len(self.neighbour_array)):
            for col, column in enumerate(self.neighbour_array[row]):
                row_array[idx] = row
                col_array[idx] = column
                grad_x_array[idx] = grad_matrix_x[row][col]  * self.weight[column]
                grad_y_array[idx] = grad_matrix_y[row][col]  * self.weight[column]
                idx += 1

        gradMxCOO  = sparse.coo_matrix( (grad_x_array, (row_array, col_array)) ).T 
        gradMyCOO  = sparse.coo_matrix( (grad_y_array, (row_array, col_array)) ).T 

        gradMx = gradMxCOO.tocsr()
        gradMy = gradMyCOO.tocsr()
        gradM2 = gradMx.dot(gradMx) + gradMy.dot(gradMy) # The del^2 operator !    
        
        if self.verbose:
            print "    Calculation of grad matrices ", time.clock()-t, "Seconds "



        ## Put these back on the mesh object     
            
        self.grad_matrix_x = grad_matrix_x
        self.grad_matrix_y = grad_matrix_y
 
        self.gradMx = gradMx
        self.gradMy = gradMy
        self.gradM2 = gradM2
        
        
        return
    
    
    
    def _matrix_store_delaunay_grad_matrix(self):
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

        NgradZx,NgradZy = self.delaunay_grad(Z)
        Ndel2Z = self.delaunay_div(NgradZx,NgradZy)
        
        gradError = (npl.norm(gradZx-NgradZx,2) + npl.norm(gradZy-NgradZy,2) ) / (npl.norm(gradZx,2) + npl.norm(gradZy,2))
        del2Error = npl.norm(del2Z-Ndel2Z,2) / npl.norm(del2Z,2)   
        
        return gradError, del2Error

        
            
    




