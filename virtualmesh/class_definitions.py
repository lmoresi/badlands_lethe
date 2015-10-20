
##
## Virtual mesh class
## This gives the structure of the meshing classes
##
##


class VirtualMesh:
    """ Abstract class for triangulated meshes and regular pixel meshes (2D)

    Outlines the methods required for the general mesh types.
    Each type of mesh must implement these or provide an elegant fallback
    """

    mesh_type="NoMesh"

    def __init__(self, **kwargs):
        print "Abstract mesh is a base mesh class and cannot be instantiated in its own right"
        raise NotImplementedError()

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


class VirtualTopoMesh(VirtualMesh):
    """ Adds methods to deal with Topography """


class VirtualSurfaceProcessMesh(VirtualTopoMesh):
    """ Adds methods to deal with erosion / transport etc """
