## Surface mesh (subclass of mesh) - this defines a TriMesh plus height plus all of the paraphernalia to evolve the height

import numpy as np
import math
from ..virtualmesh import VirtualTopoMesh

from ..petsc import coo_matrix

class TopoMesh(VirtualTopoMesh):
    """
    Builds a TriMesh (2D) object and adds a height field and data structures / operators
    to propagate information across the surface (e.g. flow downhill)
    """

    name="Generic_Height_TriMesh"

    def __init__(self):

    #    VirtualTopoMesh.__init__(self)
        super(TopoMesh, self).__init__()
        print "Topo mesh init"


    #
    #     """
    #     Initialise the Delaunay mesh (parent) and build height data structures
    #     """
    #
    #     # initialise the mesh itself from the parent TreMesh class
    #     TreMesh.__init__(self, points_x=points_x, points_y=points_y,
    #                            boundary_mask=boundary_mask,
    #                            verbose=verbose, filename=filename)
    #
    #     # Add the height field (and compute slope, create a height-sorted index list)
    #
    #     if filename:
    #         try:
    #             meshdata = np.load(filename)
    #             self.height = meshdata['height']
    #
    #         except:
    #             # Will have already bombed if not a valid mesh file
    #             print "Invalid height mesh file - ", filename
    #
    #     else:
    #         self.height = height
    #
    #     self.update_height(self.height)
    #
    #     return


    def update_height(self, height):
        """
        If the height field changes a number of things will need to be re-built. This function
        is called with the new height field which it loads onto the mesh and then builds gradients
        and sorts the nodes by height etc etc.

        It does not build / re-build the node chains as this is expensive, but it does destroy
        any existing node chains which would no longer be valid. See self.build_node_chains()
        """

        import time

        self.height = height.copy()


        gradZx, gradZy = self.derivative_grad(height)
        self.slope = np.sqrt(gradZx**2+gradZy**2)

        # Initialise the downhill/uphill data structures

        wall_time = time.clock()
        self._sort_nodes_by_height()

        if self.verbose:
            print " - Sorted all nodes by height (high to low) ", time.clock() - wall_time, "s"
            wall_time = time.clock()

        wall_time = time.clock()
        self._build_downhill_matrices()

        if self.verbose:
            print " - Built downhill matrices ", time.clock() - wall_time, "s"
            wall_time = time.clock()

        # Ensure no outdated node chain information is kept

        self.node_chain_lookup = None
        self.node_chain_list = None

    def dump_to_file(self, filename, **kwargs):
        '''
        Save HeightMesh data to a file - stores x, y and triangulation information sufficient to
        retrieve, plot and rebuild the mesh. Saves any given data

        '''

        np.savez(filename, x=self.x, y=self.y, height=self.height, bmask=self.bmask, triang=self.tri.simplices, **kwargs )


    def _sort_nodes_by_height(self):
        """
        Sort nodes from highest to lowest according to self.height
        and store as indices into height array
        """
        self.node_high_to_low = np.argsort(self.height)[::-1]

        # Also to sort neighbour node array by height

        neighbour_array_lo_hi = self.neighbour_array.copy()  # easiest way to get size / structure right

        for node in xrange(0,self.npoints):
            heights = self.height[self.neighbour_array[node]]
            neighbour_array_lo_hi[node] = self.neighbour_array[node][np.argsort(heights)]

        self.neighbour_array_lo_hi = neighbour_array_lo_hi


    def cumulative_flow(self, vector):

        DX0 = vector.copy()
        DX1 = vector.copy()

        while DX1.any():
            DX1 = self.downhillMat.dot(DX1)
            DX0 += DX1


        return DX0

    def cumulative_flow1(self, vector):

        DX0 = vector.copy()
        DX1 = vector.copy()

        while DX1.any():
            DX1 = self.adjacency1.dot(DX1)
            DX0 += DX1


        return DX0

    # def cumulative_flow16(self, vector):

    #     DX0 = vector.copy()
    #     DX1 = vector.copy()

    #     while DX1.any():
    #         DX = self.downhillMat16a.dot(DX1)
    #         DX1 = self.downhillMat16.dot(DX1)
    #         DX0 += DX

    #     return DX0

    # def cumulative_flow8(self, vector):

    #     DX0 = vector.copy()
    #     DX1 = vector.copy()

    #     while DX1.any():
    #         DX = self.downhillMat8a.dot(DX1)
    #         DX1 = self.downhillMat8.dot(DX1)
    #         DX0 += DX

    #     return DX0


    def sweep_downhill_with_flow(self, vector):

        DX0 = vector.copy()
        DX1 = vector.copy()

        while DX1.any():
            DX1 = self.downhillMat.dot(DX1)
            DX0 = self.accumulatorMat.dot(DX0)

        return DX0


## I think the best strategy here would be to build the adjacency matrix for the steepest descent
## graph and one for the second-steepest descent and store these.
##
## The actual downhill matrix can then be calculated based on the particular choice of pathway model
## (and hence downhill.T would just work as it is supposed to).
##
## It would probably be possible to make the accumulator matrix from this one.
## Not sure exactly how this would work.


    def _build_downhill_matrices(self, weight=0.6667):
        """
        Constructs a sparse matrix to move information downhill by one node (self.downhillMat).
        A second matrix (self.accumulatorMat) moves the information as far as a base level / boundary node
        where it then sticks.

        Repeated application of the accumulatorMatrix moves all information to the catchment outflow and
        can be used to identify the catchments and find the catchment area.

        The downhill matrix pushes information out of the domain and can be used as an increment
        to construct the cumulative area (etc) along the flow paths.

        Note that downhillMat.T will propagate information from the catchment outflow to every point
        in the catchment and can be used incrementally to tag nodes to find the catchments.

        """


        down_neighbour = np.empty(self.npoints, dtype=int)

        size = self.npoints
        row_array  = np.empty(size, dtype=int)
        col_array  = np.empty(size, dtype=int)
        accu_array = np.ones(size)

        # Build a matrix of downhill-ness - one entry per node !
        for node in xrange(0,self.npoints):
            down_neighbour[node] = self.neighbour_array_lo_hi[node][0]
            row_array[node] = node
            col_array[node] = down_neighbour[node]


        self.accumulatorMat = coo_matrix(row_array, col_array, accu_array, shape=(self.N,self.N), offset=(self.start,self.n), comm=self.comm).transpose()

        self._build_adjacency_matrix_1()
        self._build_adjacency_matrix_2()

        self.downhillMat = weight * self.adjacency1 + (1.0-weight) * self.adjacency2

        # A1 = self.downhillMat
        # A2  = self.downhillMat.dot(self.downhillMat)
        # A2a = A1 + A2
        # A4 = A2.dot(A2)
        # A4a = A2a + A2.dot(A2a)
        # A8 = A4.dot(A4)
        # A8a = A4a + A4.dot(A4a)
        # A16 = A8.dot(A8)
        # A16a = A8a + A8.dot(A8a)

        # self.downhillMat16  = A16
        # self.downhillMat8   = A8
        # self.downhillMat16a = A16a
        # self.downhillMat8a  = A8a

        # We make it optional to build these as they are not sparse
        # This cleans up previously stored matrices

        self.downhillCumulativeMat = None
        self.sweepDownToOutflowMat = None

        return


    def _build_adjacency_matrix_1(self):
        """
        Constructs a sparse matrix to move information downhill by one node in the
        direction of the lowest node (self.adjacency1) - NOT in the steepest direction, though
        approximately so assuming roughly equal triangle sizes

        The downhill matrix pushes information out of the domain and can be used as an increment
        to construct the cumulative area (etc) along the flow paths.

        """


        down_neighbour = np.empty(self.npoints, dtype=int)

        size = self.npoints
        row_array  = np.empty(size, dtype=int)
        col_array  = np.empty(size, dtype=int)
        down_array = np.ones(size)

        # Build a matrix of downhill-ness - one entry per node !
        for node in xrange(0,self.npoints):
            down_neighbour[node] = self.neighbour_array_lo_hi[node][0]
            row_array[node] = node
            col_array[node] = down_neighbour[node]
            if node == down_neighbour[node]:
                # Catch cases where node is local low point (i.e. it is its own low neighbour)
                down_array[node] = 0.0

        self.adjacency1 = coo_matrix(row_array, col_array, down_array, shape=(self.N,self.N), offset=(self.start,self.n), comm=self.comm).transpose()

        # Catch pathological cases - sometimes if there is a flat spot on the boundary, then
        # the filling method above will produce a non-square matrix. This is caused by
        # repetition of values in the COO list which are summed on conversion.

        if self.adjacency1.shape[0] != self.adjacency1.shape[1]:
            # This approach works but is a lot slower

            print """
            Warning: the downhill matrices require a slow build method. This is probably
            Because there are degeneracies in the slope - particularly at the boundaries
            A small random perturbation is usually enough to fix this problem
            """
            from scipy.sparse import lil_matrix
            downMat = lil_matrix((size, size))

            for row in xrange(0, self.npoints):
                downMat[down_neighbour[row],row] = 1.0

            for row in xrange(0, self.npoints):
                if down_neighbour[row] == row:
                    downMat[row,row] = 0.0

            downMat = downMat.tocoo()
            row_array, col_array, down_array = downMat.row, downMat.col, downMat.data

            self.adjacency1 = coo_matrix(row_array, col_array, down_array, shape=(self.N,self.N), offset=(self.start,self.n), comm=self.comm).transpose()

        return

    def _build_adjacency_matrix_2(self):
        """
        Constructs a sparse matrix to move information downhill by one node in the
        direction of the second-lowest node (self.adjacency2)

        The downhill matrix pushes information out of the domain and can be used as an increment
        to construct the cumulative area (etc) along the flow paths.

        """


        down_neighbour = np.empty(self.npoints, dtype=int)
        down_neighbour1 = np.empty(self.npoints, dtype=int)

        size = self.npoints
        row_array  = np.empty(size, dtype=int)
        col_array  = np.empty(size, dtype=int)
        down_array = np.ones(size)

        # Build a matrix of downhill-ness - one entry per node !
        for node in xrange(0,self.npoints):
            down_neighbour[node]  = self.neighbour_array_lo_hi[node][0]
            down_neighbour1[node] = self.neighbour_array_lo_hi[node][1]
            row_array[node] = node
            col_array[node] = down_neighbour1[node]
            if node == down_neighbour[node]:
                # Catch cases where node is local low point (i.e. it is its own low neighbour)
                down_array[node] = 0.0
            if node == down_neighbour1[node]:
                col_array[node] = down_neighbour[node]

        self.adjacency2 = coo_matrix(row_array, col_array, down_array, shape=(self.N,self.N), offset=(self.start,self.n), comm=self.comm).transpose()

        # Catch pathological cases - sometimes if there is a flat spot on the boundary, then
        # the filling method above will produce a non-square matrix. This is caused by
        # repetition of values in the COO list which are summed on conversion.

        if self.adjacency2.shape[0] != self.adjacency2.shape[1]:
            # This approach works but is a lot slower

            print """
            Warning: the downhill matrices require a slow build method. This is probably
            Because there are degeneracies in the slope - particularly at the boundaries
            A small random perturbation is usually enough to fix this problem
            """
            from scipy.sparse import lil_matrix
            downMat = lil_matrix((size, size))

            for row in xrange(0, self.npoints):
                downMat[down_neighbour[row],row] = 1.0

            for row in xrange(0, self.npoints):
                if row == down_neighbour[row] or row == down_neighbour1[row]:
                    downMat[row,row] = 0.0

            downMat = downMat.tocoo()
            row_array, col_array, down_array = downMat.row, downMat.col, downMat.data

            self.adjacency2 = coo_matrix(row_array, col_array, down_array, shape=(self.N,self.N), offset=(self.start,self.n), comm=self.comm).transpose()

        return


    def build_cumulative_downhill_matrix(self):
        """
        Build non-sparse, single hit matrices to propagate information downhill
        (self.sweepDownToOutflowMat and self.downhillCumulativeMat)

        This may be expensive in terms of storage so this is only done if
        self.storeDense == True and the matrices are also out of date (which they
        will be if the height field is changed)

        downhillCumulativeMat = I + D + D**2 + D**3 + ... D**N where N is the length of the graph

        """

        import time

        walltime = time.clock()

        downHillaccuMat = self.downhillMat.copy()
        accuM           = self.downhillMat.copy()   # work matrix

        DX =  np.ones(self.npoints) # measure when all the info has been propagated out.
        previous_nonzero = 0
        it = 0

        while np.count_nonzero(DX) != previous_nonzero:
            accuM           = accuM.dot(self.downhillMat)
            downHillaccuMat = downHillaccuMat + accuM
            previous_nonzero = np.count_nonzero(DX)

            DX = self.downhillMat.dot(DX)

            it += 1


        print " - Dense downhill matrix storage time ", time.clock() - walltime
        print " - Maximum path length ",it

        walltime = time.clock()


        # Turn this into a loop !

        A1 = self.downhillMat
        A2  = A1.dot(A1)
        A2a = A1 + A2
        A4 = A2.dot(A2)
        A4a = A2a + A2.dot(A2a)
        A8 = A4.dot(A4)
        A8a = A4a + A4.dot(A4a)
        A16 = A8.dot(A8)
        A16a = A8a + A8.dot(A8a)
        A32 = A16.dot(A16)
        A32a = A16a + A16.dot(A16a)
        A64 = A32.dot(A32)
        A64a = A32a + A32.dot(A32a)
        A128 = A64.dot(A64)
        A128a = A64a + A64.dot(A64a)

        print "A32.nnz = ", A32.nnz
        print "A64.nnz = ", A64.nnz
        print "A128.nnz = ", A128.nnz


        print " - Dense downhill matrix storage time v2", time.clock() - walltime
        print " - Maximum path length ", 128

        # make identity matrix
        size = self.npoints
        diag_IJ = np.arange(size, dtype='int32')
        diag_V  = np.ones(size, dtype='float32')

        identityMat = coo_matrix(diag_IJ, diag_IJ, diag_V, shape=(self.N,self.N), offset=(self.start,self.n), comm=self.comm)

        downHillaccuMat += identityMat
        downHillaccuMat2 = A128a + identityMat


        return downHillaccuMat, downHillaccuMat2

    def _build_cumulative_downhill_matrices(self):
        """
        Build non-sparse, single hit matrices to propagate information downhill
        (self.sweepDownToOutflowMat and self.downhillCumulativeMat)

        This may be expensive in terms of storage so this is only done if
        self.storeDense == True and the matrices are also out of date (which they
        will be if the height field is changed)

        downhillCumulativeMat = I + D + D**2 + D**3 + ... D**N where N is the length of the graph


        """

        import time

        downSweepMat    = self.accumulatorMat.copy()
        downHillaccuMat = self.downhillMat.copy()
        accuM           = self.downhillMat.copy()   # work matrix

        DX =  np.ones(self.npoints) # measure when all the info has been propagated out.

        walltime = time.clock()

        while np.any(DX):
            downSweepMat    = downSweepMat.dot(self.accumulatorMat)  # N applications of the accumulator
            accuM           = accuM.dot(self.downhillMat)
            downHillaccuMat = downHillaccuMat + accuM

            DX = self.downhillMat.dot(DX)


        print " - Dense downhill matrix storage time ", time.clock() - walltime

        # make identity matrix
        size = self.npoints
        diag_IJ = np.arange(size, dtype='int32')
        diag_V  = np.ones(size, dtype='float32')

        downHillaccuMat += coo_matrix(diag_IJ, diag_IJ, diag_V, shape=(self.N,self.N), offset=(self.start,self.n), comm=self.comm)

        self.downhillCumulativeMat = downHillaccuMat
        self.sweepDownToOutflowMat = downSweepMat

        # print "Terminated in ",it," iterations"

        return



    def _node_lowest_neighbour(self, node):
        """
        Find the lowest node in the neighbour list of the given node
        """

        lowest = self.neighbour_array_lo_hi[node][0]

        if lowest != node:
            return lowest
        else:
            return -1



    def _node_highest_neighbour(self, node):
        """
        Find the highest node in the neighbour list of the given node
        """

        highest = self.neighbour_array_lo_hi[node][-1]

        if highest != node:
            return highest
        else:
            return -1


    def _node_walk_downhill(self, node):
        """
        Walks downhill terminating when the downhill node is already claimed
        """

        chain = -np.ones(self.npoints, dtype=np.int) # in case the mesh is a spiral ziggurat

        idx = 0
        maxIdx = self.npoints
        chain[idx] = node
        low_neighbour = self._node_lowest_neighbour(node)
        junction = -1

        while low_neighbour != -1:
            idx += 1
            chain[idx] = low_neighbour
            if self.node_chain_lookup[low_neighbour] != -1:
                junction = self.node_chain_lookup[low_neighbour]
                break

            low_neighbour = self._node_lowest_neighbour(low_neighbour)

        return junction, chain[0:idx+1]


    def build_node_chains(self):
        """
        Builds all the chains for the mesh which flow from high to low and terminate
        when they meet with an existing chain.

        The following data structures become available once this function has been called:

            self.node_chain_lookup - tells you the chain in which a given node number lies
            self.node_chain_list   - is a list of the chains of nodes (each of which is an list)

        The terminating node of a chain may be the junction with another (pre-exisiting) chain
        and will be a member of that chain. Backbone chains which run from the highest level
        to the base level or the boundary are those whose terminal node is also a member of the same chain.

        Nodes which are at a base level given by self.base, are collected separately
        into chain number 0.
        """

        self.node_chain_lookup = -np.ones(self.npoints, dtype=np.int)
        self.node_chain_list = []

        node_chain_idx = 1

        self.node_chain_list.append([]) # placeholder for any isolated base-level nodes

        for node1 in self.node_high_to_low:
            if (self.node_chain_lookup[node1] != -1):
                continue

            junction, this_chain = self._node_walk_downhill(node1)

            if len(this_chain) > 1:
                self.node_chain_list.append(this_chain)

                self.node_chain_lookup[this_chain[0:-1]] = node_chain_idx
                if self.node_chain_lookup[this_chain[-1]] == -1:
                    self.node_chain_lookup[this_chain[-1]] = node_chain_idx

                node_chain_idx += 1

            else:
                self.node_chain_list[0].append(this_chain[0])
                self.node_chain_lookup[this_chain[0]] = 0

        return



## Smoothing operators (the uphill and downhill are left private since they are generally not used by themselves)

    def downhill_smoothing(self, data, its, centre_weight=0.5):

        # This could be stored as a matrix too !

        norm  = self.downhillMat.dot(np.ones_like(self.x))
        smooth_data = data.copy()

        for i in range(0,its):
            smooth_data =   (1.0-centre_weight) * self.downhillMat.dot(smooth_data)   + \
                            smooth_data * np.where(norm==0.0, 1.0, centre_weight)


        return smooth_data




    def uphill_smoothing(self, data, its, centre_weight=0.5):

        norm2 =  self.downhillMat.Tdot(np.ones_like(self.x))
        norm2[norm2 != 0.0] = 1.0 / norm2[norm2 != 0.0]
        smooth_data = data.copy()

        for i in range(0,its):
            smooth_data   = (1.0-centre_weight) * self.downhillMat.Tdot(smooth_data) * norm2  + \
                            smooth_data * np.where(norm2==0.0, 1.0, centre_weight)

        smooth_data *= data.mean() / smooth_data.mean()

        return smooth_data


    def streamwise_smoothing(self, data, its, centre_weight=0.5):
        """
        A smoothing operator that is limited to the uphill / downhill nodes for each point. It's hard to build
        a conservative smoothing operator this way since "boundaries" occur at irregular internal points associated
        with watersheds etc. Upstream and downstream smoothing operations bracket the original data (over and under,
        respectively) and we use this to find a smooth field with the same mean value as the original data. This is
        done for each application of the smoothing.

        """

        smooth_data_d = self.downhill_smoothing(data, its, centre_weight=centre_weight)
        smooth_data_u = self.uphill_smoothing(data, its, centre_weight=centre_weight)



        return 0.5 * (smooth_data_d + smooth_data_u)



    def _matrix_store_smooth_downhill(self):
        """
        Creates the sparse matrix form of the downhill operator.

        """

        import time


        t = time.clock()


        size = 0
        for nl in self.neighbour_array_lo_hi:
            size += 3 # len(nl)

        row_array = np.empty(size, dtype = int)
        col_array = np.empty(size, dtype = int)
        slope_array = np.zeros(size)
        local_slope_array = np.zeros(64)


        idx=0
        for row in range(0, len(self.neighbour_array_lo_hi)):
            neighbours  = self.neighbour_array_lo_hi[row]
            npoints  = self.points[neighbours]

            ## work out (downhill) gradient to (max of three) nearby neighbours


            for col, column in enumerate(neighbours[0:3]):

                delta_h = self.height[column] - self.height[row]


                if delta_h < 0.0:
                    delta_s2  = (self.x[column] - self.x[row])**2 + (self.y[column] - self.y[row])**2
                    local_slope_array[col] = ( delta_h**2 / delta_s2 )**5

                elif delta_h == 0.0 and self.bmask[row] == False:
                    local_slope_array[col] = 1.0e-20

                else:
                    local_slope_array[col] = 1.0e-20

           # Normalise this so that it conserves mass (note - low points will have no contributions here !)

            norm = local_slope_array[0:len(neighbours)].sum()
            if norm != 0.0:
                norm = 1.0 / norm

            for col, column in enumerate(neighbours[0:3]):
                row_array[idx] = row
                col_array[idx] = column
                slope_array[idx] = local_slope_array[col] * norm

                idx += 1

        # We can re-pack this array into a sparse matrix for v. fast computation of downhill operator
        slopeMat = coo_matrix(row_array, col_array, slope_array, shape=(self.N,self.N), offset=(self.start,self.n), comm=self.comm).transpose()

        print "SlopeMat.shape ", slopeMat.shape, size

    #     slopeNormVec = np.array(slopeMat.sum(axis=1)).T[0]
    #     slopeNormVec[slopeNormVec != 0.0] = 1.0 / slopeNormVec[slopeNormVec != 0.0]
    #     slopeNormMat = sparse.eye(self.npoints)
    #     slopeNormMat.setdiag(slopeNormVec)
    #     slopeMat = slopeNormMat.dot(slopeMat)

        self.smoothDownhillMat = slopeMat

        return
