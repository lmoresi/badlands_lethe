## Surface mesh (subclass of mesh) - this defines a TriMesh plus height plus all of the paraphernalia to evolve the height


import numpy as np
import math
from ..virtualmesh import VirtualSurfaceProcessMesh

class SurfaceProcessMesh(VirtualSurfaceProcessMesh):
    """
    Builds a TriMesh/HeightMesh (2D) object and adds a height field and data structures / operators
    to propagate information across the surface (e.g. flow downhill)
    """

    name="Generic_SurfaceProcess_TriMesh"

    def __init__(self, **kwargs):
        super(SurfaceProcessMesh, self).__init__()
        print "Surface mesh init"


    def update_surface_processes(self, rainfall_pattern, sediment_distribution):

        self.rainfall_pattern = rainfall_pattern.copy()
        self.sediment_distribution = sediment_distribution.copy()

        import time

        # Build the chains of down-ness

        # Find the catchments

        wall_time = time.clock()
        # self.identify_catchments()

        # if self.verbose:
        #     print " - Built Catchment membership ", time.clock() - wall_time, "s"


        # Calculate upstream area

        wall_time = time.clock()
        # self.calc_upstream_area()

        # if self.verbose:
        #     print " - Built Upstream Area ", time.clock() - wall_time, "s"


        # Calculate distance to outflow

        wall_time = time.clock()
        # self.calc_distance_to_outflow()

        # if self.verbose:
        #     print " - Built Outflow distances ", time.clock() - wall_time, "s"
        #     wall_time = time.clock()

        wall_time = time.clock()
        self.upstream_area = self.cumulative_flow(self, self.area)

        if self.verbose:
            print " - Built cumulative area", time.clock() - wall_time, "s"
            wall_time = time.clock()

        wall_time = time.clock()
        self.low_points = self.identify_low_points()

        if self.verbose:
            print " - Found low points", time.clock() - wall_time, "s"
            wall_time = time.clock()


        wall_time = time.clock()
        self.outflow_points = self.identify_outflow_points()

        if self.verbose:
            print " - Found outflow points", time.clock() - wall_time, "s"
            wall_time = time.clock()



        return


    def dump_to_file(self, filename, **kwargs):
        '''
        Save SurfaceProcessMesh data to a file - stores x, y, bmask, height, rainfall, sediment
        and triangulation information sufficient to
        retrieve, plot and rebuild the mesh. Saves any given data as well.

        '''

        np.savez(filename, x=self.x, y=self.y, height=self.height,
                           rainfall_pattern=self.rainfall_pattern, sediment=self.sediment,
                           bmask=self.bmask, triang=self.tri.simplices, **kwargs )


    def identify_catchments_from_chains(self):
        """
        Define the catchment to which any given node belongs and store as
        self.node_catchments. Build a list of all the catchments found
        and store the backbone chain to identify them.
        Note: Catchment 0 is reserved for all those nodes which are at the base level

        """

        self.node_catchments = -np.ones(self.tri.npoints, dtype=int)
        self.catchment_list = [0]
        catchment = 1

        # node_chain_list[0] is a collection of the base-level isolated nodes
        # Catchment zero holds all of these

        for idx, chain in enumerate(self.node_chain_list[1:], start=1):
            if self.node_chain_lookup[chain][0] == self.node_chain_lookup[chain][-1]:
                self.catchment_list.append(self.node_chain_lookup[chain][0])
                self.node_catchments[chain] = catchment
                catchment += 1

        # print "Chains defining catchments --> ", catchment_list

        # deal with catchment 0 nodes
        self.node_catchments[self.node_chain_list[0]] = 0

        ## Work progressively up the ordering of the chains
        ## - find all the nodes associated with a particular catchment
        ## The termination condition will be that all nodes / chains are claimed.

        loops = 0
        while np.count_nonzero(self.node_catchments == -1) and loops < 20:
            # Search for all cases where the terminating node is order-1
            for idx, chain in enumerate(self.node_chain_list[1:]):
                if self.node_catchments[chain[0]] == -1 and self.node_catchments[chain[-1]] != -1:
                    self.node_catchments[chain[0:-1]] = self.node_catchments[chain[-1]]

            loops += 1

        if np.count_nonzero(self.node_catchments == -1) != 0:
            print "Nodes were not identified into catchments !"
            print "Loops - ", loops
            print np.where(self.node_catchments == -1)

        return


    def calc_upstream_area_along_chains(self):
        """
        Build an array (self.node_upstream_area) that is the contributing area of
        all the upstream nodes to the one in question.
        """
        self.node_upstream_area = np.copy(self.area)

        for chain in self.node_chain_list[::-1]:

            for nn, node in enumerate(chain[0:-1]):
                self.node_upstream_area[chain[nn+1]] += self.node_upstream_area[node]
               #  print "chain: ", chain_idx, "node ", chain[nn+1]," += node ",node, " / ",  node_upstream_area[node]

        return


    def calc_distance_to_outflow_along_chains(self):
        """
        Build an array (self.node_downstream_distance) that is the
        along-stream distance for any given node to the eventual outflow.
        """

        self.node_downstream_distance = np.zeros_like(self.x)

        for chain in self.node_chain_list[1:-1]:
            launch_node = chain[-1]
            for nn, node in enumerate(chain[-2::-1]):
                deltaS = math.sqrt((self.x[node] - self.x[launch_node])**2 + (self.y[node] - self.y[launch_node])**2 )
                self.node_downstream_distance[node] = self.node_downstream_distance[launch_node] + deltaS
                launch_node = node


        return

    def handle_low_points(self, base, its, verbose=False):
        """
        If the mesh has local minima and only erosion is active then it is necessary
        to do something about these local low points. Here what we do is to fill back
        upstream from the next-lowest height.

        The approach in this subroutine is independent of the available sediment so it
        does not properly conserve mass so you have to do something about this yourself !
        """

        self.low_points = self.identify_low_points()

        if len(self.low_points) == 0:
            return(self.height)

        rejected = 0
        fixed = 0
        notfixed = 0
        fix_path_nodes = 0

        delta_height = np.zeros_like(self.height)

        for point in self.low_points:
            if self.height[point] - base < 0.005 * (self.height.max() - base):
                rejected += 1
                continue

            # find the mean height in the neighbourhood and fill up everything nearby
            fixed += 1
            delta_height[point] = self.height[self.neighbour_array_lo_hi[point]].mean()
            if verbose:
                print point, "Old h", self.height[point], "->", delta_height[point]


        # Now march the new height to all the uphill nodes of these nodes

        height = np.maximum(self.height, delta_height)

        for p in range(0, its):
            delta_height = 1.001 * self.adjacency1.T.dot(delta_height)
            height = np.maximum(height, delta_height)

        if verbose:
            print "Updated", fixed, "points"
            print "Rejected ", rejected," points close to the base level"

        return height


    def identify_flat_spots(self):

        smooth_grad1 = self.local_area_smoothing(self.slope, its=1, centre_weight=0.5)
        flat_spot_field = np.where(smooth_grad1 < smooth_grad1.max() / 10, 0.0, 1.0)
        flat_spots = np.where(smooth_grad1 < smooth_grad1.max() / 10, True, False)

        return flat_spots




    def identify_low_points(self):
        """
        Identify if the mesh has (internal) local minima and return an array of node indices
        """

        low_point_list = []
        for node in range(0,self.tri.npoints):
            if self.neighbour_array_lo_hi[node][0] == node and self.bmask[node] == True:
                low_point_list.append(node)


        return np.array(low_point_list)

    def identify_high_points(self):
        """
        Identify if the mesh has (internal) local minima and return an array of node indices
        """

        high_point_list = []
        for node in range(0,self.tri.npoints):
            if self.neighbour_array_lo_hi[node][-1] == node and self.bmask[node] == True:
                high_point_list.append(node)


        return np.array(high_point_list)


    def identify_outflow_points(self):
        """
        Identify the (boundary) outflow points and return an array of node indices
        """

        outflow_point_list = []
        for node in range(0,self.tri.npoints):
            if self.neighbour_array_lo_hi[node][0] == node and self.bmask[node] == False:
                outflow_point_list.append(node)

        return np.array(outflow_point_list)



## A simple implementation of the stream power erosion rate which assumes no variation in erodability (efficiency)


    def stream_power_erosion_deposition_rate(self, efficiency=0.01,
             smooth_power=3, smooth_low_points=3, smooth_erosion_rate=3, smooth_deposition_rate = 3):
        """
        Function of the SurfaceProcessMesh which computes stream-power erosion and deposition rates
        from a given rainfall pattern (self.rainfall_pattern).

        In this model we assume a the carrying capacity of the stream is related to the stream power and so is the
        erosion rate. The two are related to one another in this particular case by a single contant (everywhere on the mesh)
        This does not allow for spatially variable erodability and it does not allow for differences in the dependence
        of erosion / deposition on the stream power.

        Deposition occurs such that the upstream-integrated eroded sediment does not exceed the carrying capacity at a given
        point. To conserve mass, we have to treat internal drainage points carefully and, optionally, smooth the deposition
        upstream of the low point. We also have to be careful when stream-power and carrying capacity increase going downstream.
        This produces a negative deposition rate when the flow is at capacity. We suppress this behaviour and balance mass across
        all other deposition sites but this does mean the capacity is not perfectly satisfied everywhere.

        parameters:
         efficiency=0.01          : erosion rate for a given stream power compared to carrying capacity
         smooth_power=3           : upstream / downstream smoothing of the stream power (number of cycles of smoothing)
         smooth_low_points=10     : upstream smoothing of the deposition at low points (number of cycles of smoothing)
         smooth_erosion_rate=0    : upstream / downstream smoothing of the computed erosion rate (number of cycles of smoothing)
         smooth_deposition_rate=0 : upstream / downstream smoothing of the computed erosion rate (number of cycles of smoothing)

        """

    # Calculate stream power

        rainflux = self.rainfall_pattern
        rainfall = self.area * rainflux
        cumulative_rain = self.cumulative_flow(rainfall)
        cumulative_flow_rate = cumulative_rain / self.area
        stream_power = self.streamwise_smoothing(cumulative_flow_rate * self.slope, smooth_power)

        if len(self.low_points):
            stream_power[self.low_points] = 0.0     # Otherwise mass disappears ...

    #  predicted erosion rate from stream power * efficiency
    #  maximum sediment that can be transported is limited by the local carrying capacity (assume also prop to stream power)
    #  whatever cannot be passed on has to be deposited

        erosion_rate = self.streamwise_smoothing(efficiency * stream_power, smooth_erosion_rate)
        full_capacity_sediment_flux = stream_power
        full_capacity_sediment_load = stream_power * self.area
        cumulative_eroded_material = self.cumulative_flow(self.area * erosion_rate)

    # But this can exceed the carrying capacity

        transport_limited_eroded_material = np.minimum(cumulative_eroded_material, full_capacity_sediment_load)
        transport_limited_erosion_rate = transport_limited_eroded_material / self.area

    # And this therefore implies a deposition rate which reduces the total sediment in the system to capacity
    # Calculate this by substracting the deposited amounts from the excess integrated flow. We could then iterate
    # to compute the new erosion rates etc, but here we just spread the sediments around to places where
    # the deposition is positive

        excess = cumulative_eroded_material - transport_limited_eroded_material
        deposition = excess - self.downhillMat.dot(excess)
        deposition2 = deposition.copy()
        deposition = np.clip(deposition, 0.0, 1.0e99)


        deficit = deposition2 - deposition
        deposit_points = np.where( deficit > 0.0 )
        deposition[deposit_points[0]] += deficit.sum() / len(deposit_points[0])

    # The (interior) low points are a bit of a problem - we stomped on the stream power there
    # but this produces a very lumpy deposition at the low point itself and I think this could
    # make the numerical representation pretty unstable. Instead what we can do is to take that
    # deposition at the low points and push it back upstream via the smoothing operator
    # I am unsure how much to do this so it is a free parameter at the moment.

        if len(self.low_points):
            low_point_deposition = np.zeros_like(deposition)
            low_point_deposition[self.low_points] = deposition[self.low_points]
            low_point_deposition = self.streamwise_smoothing(low_point_deposition, smooth_low_points)

            deposition[self.low_points] = 0.0
            deposition += low_point_deposition

        deposition_rate = self.streamwise_smoothing(deposition / self.area, smooth_deposition_rate)

        return erosion_rate, deposition_rate, stream_power


## This is the one we are currently using !!

    def stream_power_erosion_deposition_rate2(self, efficiency=0.01, smooth_power=3, \
                                              smooth_low_points=2, smooth_erosion_rate=2, \
                                              smooth_deposition_rate=2, smooth_operator=None,
                                              centre_weight_u=0.5, centre_weight=0.5):

        """
        Function of the SurfaceProcessMesh which computes stream-power erosion and deposition rates
        from a given rainfall pattern (self.rainfall_pattern).

        In this model we assume a the carrying capacity of the stream is related to the stream power and so is the
        erosion rate. The two are related to one another in this particular case by a single contant (everywhere on the mesh)
        This does not allow for spatially variable erodability and it does not allow for differences in the dependence
        of erosion / deposition on the stream power.

        Deposition occurs such that the upstream-integrated eroded sediment does not exceed the carrying capacity at a given
        point. To conserve mass, we have to treat internal drainage points carefully and, optionally, smooth the deposition
        upstream of the low point. We also have to be careful when stream-power and carrying capacity increase going downstream.
        This produces a negative deposition rate when the flow is at capacity. We suppress this behaviour and balance mass across
        all other deposition sites but this does mean the capacity is not perfectly satisfied everywhere.

        parameters:
         efficiency=0.01          : erosion rate for a given stream power compared to carrying capacity
         smooth_power=3           : upstream / downstream smoothing of the stream power (number of cycles of smoothing)
         smooth_low_points=3      : upstream smoothing of the deposition at low points (number of cycles of smoothing)
         smooth_erosion_rate=0    : upstream / downstream smoothing of the computed erosion rate (number of cycles of smoothing)
         smooth_deposition_rate=0 : upstream / downstream smoothing of the computed erosion rate (number of cycles of smoothing)

        """


        if smooth_operator == None:
            smooth_operator = self.streamwise_smoothing

    # Calculate stream power

        rainflux = self.rainfall_pattern
        rainfall = self.area * rainflux
        cumulative_rain = self.cumulative_flow(rainfall)
        cumulative_flow_rate = cumulative_rain / self.area
        stream_power = self.uphill_smoothing(cumulative_flow_rate * self.slope, smooth_power, centre_weight=centre_weight_u)

        # if len(self.low_points):
        #     stream_power[self.low_points] = 0.0     # Otherwise mass disappears ...

    #  predicted erosion rate from stream power * efficiency
    #  maximum sediment that can be transported is limited by the local carrying capacity (assume also prop to stream power)
    #  whatever cannot be passed on has to be deposited

        erosion_rate = self.streamwise_smoothing(efficiency * stream_power, smooth_erosion_rate, centre_weight=centre_weight)
        full_capacity_sediment_flux = stream_power
        full_capacity_sediment_load = stream_power * self.area
        cumulative_eroded_material = self.cumulative_flow(self.area * erosion_rate)

    # But this can exceed the carrying capacity

        transport_limited_eroded_material = np.minimum(cumulative_eroded_material, full_capacity_sediment_load)
        transport_limited_erosion_rate = transport_limited_eroded_material / self.area

    # And this therefore implies a deposition rate which reduces the total sediment in the system to capacity
    # Calculate this by substracting the deposited amounts from the excess integrated flow. We could then iterate
    # to compute the new erosion rates etc, but here we just spread the sediments around to places where
    # the deposition is positive

        excess = cumulative_eroded_material - transport_limited_eroded_material
        deposition = excess - self.downhillMat.dot(excess)
        depo_sum = deposition.sum()


    # Now rebalance the fact that we have clipped off the negative deposition which will need
    # to be clawed back downstream (ideally, but for now we can just make a global correction)

        deposition = np.clip(deposition, 0.0, 1.0e99)
        deposition *= depo_sum / (deposition.sum() + 1.0e-12)


    # The (interior) low points are a bit of a problem - we stomped on the stream power there
    # but this produces a very lumpy deposition at the low point itself and this could (does)
    # make the numerical representation pretty unstable. Instead what we can do is to take that
    # deposition at the low points let it spill into the local area


    ## These will instead be handled by a specific routine "handle_low_points" which is
    ## done once the height has been updated

        if len(self.low_points):
            deposition[self.low_points] = 0.0

    # The flat regions in the domain are also problematic since the deposition there is

        flat_spots = self.identify_flat_spots()

        if len(flat_spots):
            smoothed_deposition = deposition.copy()
            smoothed_deposition[np.invert(flat_spots)] = 0.0
            smoothed_deposition = self.local_area_smoothing(smoothed_deposition, its=2, centre_weight=0.5)
            deposition[flat_spots] = smoothed_deposition[flat_spots]

        deposition_rate = smooth_operator(deposition , smooth_deposition_rate, centre_weight=centre_weight) / self.area

        return erosion_rate, deposition_rate, stream_power



    def landscape_diffusion_critical_slope(self, kappa, critical_slope, fluxBC):
        '''
        Non-linear diffusion to keep slopes at a critical value. Assumes a background
        diffusion rate (can be a vector of length mesh.tri.npoints) and a critical slope value.

        This term is suitable for the sloughing of sediment from hillslopes.

        To Do: The critical slope should be a function of the material (sediment, basement etc)
        but currently it is not.

        To Do: The fluxBC flag is global ... it should apply to the outward normal
        at selected nodes but currently it is set to kill both fluxes at all boundary nodes.
        '''

        inverse_bmask = np.invert(self.bmask)

        kappa_eff = kappa / (1.01 - (np.clip(self.slope,0.0,critical_slope) / critical_slope)**2)
        diff_timestep   =  self.area.min() / kappa_eff.max()


        gradZx, gradZy = self.delaunay_grad(self.height)
        flux_x = kappa_eff * gradZx
        flux_y = kappa_eff * gradZy
        if fluxBC:
            flux_x[inverse_bmask] = 0.0
            flux_y[inverse_bmask] = 0.0  # outward normal flux, actually
        diffDz  = self.delaunay_div(flux_x, flux_y)

        if not fluxBC:
            diffDz[inverse_bmask] = 0.0

        return diffDz, diff_timestep
