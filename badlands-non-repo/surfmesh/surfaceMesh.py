## Surface mesh (subclass of mesh) - this defines a TriMesh plus height plus all of the paraphernalia to evolve the height
    

import numpy as np
import math
from .. import TreMesh
from . import HeightMesh 

class SurfaceProcessMesh(HeightMesh):
    """
    Builds a TriMesh/HeightMesh (2D) object and adds a height field and data structures / operators
    to propagate information across the surface (e.g. flow downhill) 
    """
    
    name="Generic_SurfaceProcess_TriMesh"

    
    def __init__(self, points_x, points_y, height, rainfall_pattern, boundary_mask, verbose=False, storeDense=False):
        """
        Initialise the Delaunay mesh (parent) and build height data structures
        """
        
        # initialise the mesh itself from the parent HeightMesh class

        HeightMesh.__init__(self, points_x, points_y, height, boundary_mask, verbose=verbose, storeDense=storeDense)
               
        # From the height field, build the river networks etc

        self.rainfall_pattern = rainfall_pattern
        self.sediment_thickness = np.zeros_like(rainfall_pattern)

        self.update_surface_processes()   
       
        return

    def update_surface_processes(self):
    
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
        self.upstream_area = HeightMesh.cumulative_flow(self, self.area)

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

    # def handle_low_points(self, base, its):
    #     """
    #     If the mesh has local minima and only erosion is active then it is necessary
    #     to do something about these local low points. Here we assume that we can assign
    #     the local average height to any local minimum and then iterate to get them to go 
    #     away.  This is entirely ad hoc !
    #     """

    #     for iteration in range(0,its):
    #         low_points = 0
    #         for node in range(0,self.tri.npoints):
    #             if self.neighbour_array_lo_hi[node][0] == node and self.bmask[node] == True:                   
    #                 self.height[node] = self.height[self.neighbour_list[node]].mean()
    #                 low_points += 1

    #         if low_points == 0:
    #             break 

    def handle_low_points(self, base, its):
        """
        If the mesh has local minima and only erosion is active then it is necessary
        to do something about these local low points. Here we brute force our way
        from the low point to some lower point nearby being careful not to create a
        loop in the process. If the height is changing progressively, then this Should
        find a nearby neighbour once removed from any point, but in practice it can be 
        necessary to search further. If the region is almost flat then this process 
        is pointless as our assumption of equilibrated downhill flow is false.
        """
        
        rejected = 0
        fixed = 0
        notfixed = 0
        fix_path_nodes = 0

        delta_height = np.zeros_like(self.height)
        
        for point in self.low_points:
     
            # Flat areas + dominated by deposition ... would be a better test !
            
            if self.height[point]-base < 0.005 * (self.height.max()-base):
                rejected += 1
                continue
               
            loops = 0   
            chain = [point]
            while len(chain) < its and loops < 2*its:
                loops += 1
                for node in self.neighbour_array_lo_hi[chain[-1]]:
                    if node not in chain:
                        chain.append(node)
                        break
                
                if self.height[chain[-1]] < self.height[point]:
                    break
                       
            # Case where this worked:
            if self.height[chain[-1]] - self.height[chain[0]] < 0.0:
                # print point, " ---> ",mesh5.height[point], self.height[next_lowest], self.height[connect_to], self.height[point]- self.height[connect_to]
                fixed = fixed + 1
                fix_path_nodes = max(fix_path_nodes, len(chain))
                
                ddelta = self.height[chain[0]] - self.height[chain[-1]]
                delta = ddelta / (len(chain) - 1)
                
                for idx, node in enumerate(chain):
                    delta_height[node] =  - idx * delta
            
                            
            else:
                notfixed = notfixed + 1
                # print point, " -/-> ",self.height[point], self.height[next_lowest], self.height[connect_to], self.height[point]- self.height[connect_to]
                # print point, "      ",point, next_lowest, connect_to
         

        delta_height = self.local_area_smoothing(delta_height, 5 , centre_weight=0.9)

        self.height += delta_height


        report_string = "Low points - {:d} points not considered, {:d} fixed ({:d}), {:d} couldn't be fixed ".format(rejected, fixed, fix_path_nodes, notfixed)
        
        return report_string


    def handle_low_points2(self, base, its):
        """
        If the mesh has local minima and only erosion is active then it is necessary
        to do something about these local low points. Here we brute force our way
        from the low point to some lower point nearby being careful not to create a
        loop in the process. If the height is changing progressively, then this Should
        find a nearby neighbour once removed from any point, but in practice it can be 
        necessary to search further. If the region is almost flat then this process 
        is pointless as our assumption of equilibrated downhill flow is false.
        """
        
        rejected = 0
        fixed = 0
        notfixed = 0
        fix_path_nodes = 0

        delta_height = np.zeros_like(self.height)
        
        for point in self.low_points:
            fixed += 1
            if self.height[point]-base < 0.005 * (self.height.max()-base):
                rejected += 1
                continue


            # find the next lowest point in the neighbourhood and fill up everything nearby
            
            delta_height[point] = self.neighbour_array_lo_hi[point][1]
    
           
        # Now march the new height to all the uphill nodes of these nodes

        for p in range(0, its):
            delta_height += 1.0001 * self.uphill_smoothing(delta_height, its=1, centre_weight=0.0) # tiny gradient

        self.height = np.maximum(self.height, delta_height)

        print "dH min", delta_height.min()
        print "dH max", delta_height.max()


        report_string = "Low points - {:d} points not considered, {:d} fixed ({:d}), {:d} couldn't be fixed ".format(rejected, fixed, fix_path_nodes, notfixed)
        
        return report_string





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


    def stream_power_erosion_deposition_rate(self, efficiency=0.01, smooth_power=3, smooth_low_points=3, smooth_erosion_rate=3, smooth_deposition_rate = 3):
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
        deposition *= depo_sum / deposition.sum()


    # The (interior) low points are a bit of a problem - we stomped on the stream power there
    # but this produces a very lumpy deposition at the low point itself and this could (does)
    # make the numerical representation pretty unstable. Instead what we can do is to take that
    # deposition at the low points let it spill into the local area
            
        # if len(self.low_points):
        #     low_point_deposition = np.zeros_like(deposition)
        #     low_point_deposition[self.low_points] = deposition[self.low_points]

        #     #for i in range(0, smooth_low_points):
        #         # low_point_deposition = self.local_area_smoothing(low_point_deposition, 1, centre_weight=0.75)  
        #     low_point_deposition = self.uphill_smoothing(low_point_deposition, smooth_low_points, centre_weight=0.33) 

        #     deposition[self.low_points] = 0.0
        #     deposition += low_point_deposition

        
        # deposition *= depo_sum / deposition.sum()


        deposition_rate = smooth_operator(deposition , smooth_deposition_rate, centre_weight=centre_weight) / self.area

        return erosion_rate, deposition_rate, stream_power



    def landscape_diffusion_critical_slope(mesh, kappa, critical_slope, fluxBC):
        
        inverse_bmask = np.invert(mesh.bmask)
        
        kappa_eff = kappa / (1.01 - (np.clip(mesh.slope,0.0,critical_slope) / critical_slope)**2)
        diff_timestep   =  mesh.area.min() / kappa_eff.max()

        ## Should wrap this as grad ( A * grad )
        
        gradZx, gradZy = mesh.delaunay_grad(mesh.height)   
        flux_x = kappa_eff * gradZx
        flux_y = kappa_eff * gradZy    
        if fluxBC:
            flux_x[inverse_bmask] = 0.0
            flux_y[inverse_bmask] = 0.0  # outward normal flux, actually 
        diffDz  = mesh.delaunay_div(flux_x, flux_y)
        
        if not fluxBC:
            diffDz[inverse_bmask] = 0.0
        
        return diffDz, diff_timestep





   
