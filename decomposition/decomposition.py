import numpy as np
from ..virtualmesh import VirtualDecomposition

from mpi4py import MPI
default_comm = MPI.COMM_WORLD

class MeshDecomposition(VirtualDecomposition):
    """
    Mesh can be decomposed in 3 ways:
        1. Radial decomposition - each processor receives a slice of pie
        2. Ring decomposition - each processor receives a ring
        3. Grid decomposition - each proc gets a rectangle
        4. Custom decomposition - provide a condition to each processor e.g. points_x > 0


    This class divides up the mesh most efficiently, so each processor receives a
    chunk of equal size (or close to equal depending on the number of nodes/CPUs).
    build_mesh() is initiated by each processor post-decomposition, the local points
    can be accessed by self.x, self.y, self.bmask as normal.

    Problems that require just one processor need not bother with decomposition
    and can use build_mesh() directly.


        ATTRIBUTES
            comm                  : MPI communicator object
            N                     : global size of the domain
            n                     : local size on each processor
            start                 : offset for local processor
            identity              : map of which points belong to each processor
            decomposition_type    : type of decomposition (radial/ring/etc)

    """
    def __init__(self):
        """
        Initialise the class
        """

        super(MeshDecomposition, self).__init__()

        self.decomposition_type = "no decomposition"
        self.comm = default_comm

    def _decomposition_structures(self, global_size, comm=None):
        """
        Allocate local sizes to processors.

            ARGUMENTS
                global_size       : total number of points within the domain
                comm (optional)   : MPI communicator object
        """
        self.N = global_size

        if comm != None:
            self.comm = comm # MPI communicator
        self.identity = np.zeros(global_size, dtype=int)

        self.n = global_size // comm.size + int(comm.rank < (global_size % comm.size))
        self.start = 0
        for r in xrange(0, comm.rank):
            self.start += global_size // comm.size + int(r < (global_size % comm.size))


    def radial_decomposition(self, global_x, global_y, boundary_mask):
        """
        Initialise global mesh components.

            ARGUMENTS
                global_x          : global nodes in the x coordinate plane
                global_y          : global nodes in the y coordinate plane
                boundary_mask     : boolean mask of boundary points

        Divides the mesh like a slice of pie.
        Works best for circular domains.
        """
        self._decomposition_structures(len(global_x))
        self.decomposition_type = "radial decomposition"

        global_x = np.array(global_x)
        global_y = np.array(global_y)
        boundary_mask = np.array(boundary_mask)


        # centroid
        rx = global_x - global_x.mean()
        ry = global_y - global_y.mean()
        order = np.arctan2(rx, ry).argsort()

        ownership_range = np.array_split(order, self.comm.size)
        for r, ownership in enumerate(ownership_range):
            self.identity[ownership] = r

        # Have some of the pie
        local_range = ownership_range[self.comm.rank]

        local_x = global_x[local_range]
        local_y = global_y[local_range]
        local_bmask = boundary_mask[local_range]

        # Build local mesh
        self.build_mesh(local_x, local_y, local_bmask)


    def ring_decomposition(self, global_x, global_y, boundary_mask):
        """
        Divides the mesh into ring segments.
        Works best for circular domains
        """
        self._decomposition_structures(len(global_x))
        self.decomposition_type = "ring decomposition"

        # centroid
        rx = points_x - points_x.mean()
        ry = points_y - points_y.mean()
        order = np.hypot(rx, ry).argsort()

        ownership_range = np.array_split(order, self.comm.size)
        for r, ownership in enumerate(ownership_range):
            self.identity[ownership] = r

        # Have a ring segment
        local_range = ownership_range[self.comm.rank]

        local_x = points_x[local_range]
        local_y = points_y[local_range]

        # Build local mesh
        self.build_mesh(local_x, local_y, boundary_mask)


    def grid_decomposition(self, global_x, global_y, boundary_mask):
        """
        Divides the mesh into grids.
        Works best for rectangular domains.
        """
        self._decomposition_structures(len(global_x))
        self.decomposition_type = "grid decomposition"



    def custom_decomposition(self, global_x, global_y, boundary_mask, condition):
        self._decomposition_structures(len(global_x))
        self.decomposition_type = "custom decomposition"
