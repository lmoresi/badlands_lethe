
##
## Virtual mesh class
## This gives the structure of the meshing classes
##
##

class VirtualDecomposition(object):
    """
    Abstract class for mesh decomposition.
    Provides fallbacks for methods that do not exist.
    """
    def __init__(self):
        """
        Generic initialisation for decomposition class
        """

        print "VirtualDecomposition __init__fn"

        self.decomposition_type = None
        self.comm = None
        self.identity = None
        self.N = None
        self.n = None
        self.start = None

    def radial_decomposition(self, global_x, global_y, boundary_mask, **kwargs):
        raise NotImplementedError()

    def ring_decomposition(self, global_x, global_y, boundary_mask, **kwargs):
        raise NotImplementedError()

    def grid_decomposition(self, global_x, global_y, boundary_mask, **kwargs):
        raise NotImplementedError()

    def custom_decomposition(self, global_x, global_y, boundary_mask, **kwargs):
        raise NotImplementedError()


class VirtualMesh(VirtualDecomposition):
    """ Abstract class for triangulated meshes and regular pixel meshes (2D)

    Outlines the methods required for the general mesh types.
    Each type of mesh must implement these or provide an elegant fallback
    """

    def __init__(self):
        """ Generic initialisation for non-specific
            Mesh implementation
        """

        super(VirtualMesh, self).__init__()
        ## This is a place to list variables that need to
        ## be defined on the class

        print "VirtualMesh __init__fn"

        self.mesh_type = None
        self.verbose = False

        pass


    def build_mesh(self, **kwargs):
        # Absorb any args in order to raise the correct exception !
        raise NotImplementedError()

    def copy(self, **kwargs):
        """ Must copy sufficient information to be able to update / rebuild mesh"""
        raise NotImplementedError()

    def read_from_file(self, **kwargs):
        raise NotImplementedError()

    def write_to_file(self, **kwargs):
        raise NotImplementedError()

    def node_neighbours(self, centre_point):
        raise NotImplementedError()

    def build_neighbours(self):
        raise NotImplementedError()

    def build_node_weights_and_measures(self):
        raise NotImplementedError()

    def interpolate(self, data, coord, error_value=0.0):
        """ Interpolate node data array to coord(s) in coord list

        Error value is returned if the coord is out of the mesh
        """
        raise NotImplementedError()

    def derivative_grad(self, PHI):
        """ Gradient of scalar data, PHI, on the mesh.
        """
        raise NotImplementedError()

    def derivative_grad(self, PSIx, PSIy):
        """ Divergence of vector data, PSIx, PSIy, on the mesh.
        """
        raise NotImplementedError()

    def derivative_del2(self, PHI):
        """ Laplacian of scalar data, PHI, on the mesh.
        """
        raise NotImplementedError()

    ## Consider adding PLOT methods

    ## Consider adding smoothing kernels


## These need to reflect the correct names, but the topomesh and surfmesh files will still work as expected.

class VirtualTopoMesh(VirtualMesh):
    """ Adds methods to deal with Topography """

    def __init__(self):

        super(VirtualTopoMesh, self).__init__()
        # print "VirtualTopoMesh __init__ fn"

        ## Placeholders for variables which need to be set
        self.height = None
        self.slope = None
        self.node_high_to_low = None
        self.neighbour_array_lo_hi = None

    def mesh_update_height():
        pass

class VirtualSurfaceProcessMesh(VirtualTopoMesh):
    """ Adds methods to deal with erosion / transport etc """

    def __init__(self):

        super(VirtualSurfaceProcessMesh, self).__init__()
        print "VirtualSPMesh __init__ fn"

        ## Placeholders for variables which need to be set
        self.rainfall_pattern = None


    def sp_specific_methods():
        pass
