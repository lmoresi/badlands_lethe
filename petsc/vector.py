import numpy as np
import sys, petsc4py
from petsc4py import PETSc
petsc4py.init(sys.argv)

class Vector(PETSc.Vec):
    """
    Extends the PETSc.Vec class with additional methods and simple set up.

    The PETSc.Vec class is much more complete than the PETSc.Mat class.
    Methods and attributes are similar to a NumPy ndarray.
    """

    name = "PETSc_Vector"

    def __init__(self, data, size, comm):
        """
        Initialises the PETSc Vector object.

            ARGUMENTS
                data  : chunk of data on a processor
                size  : global size of the vector
                comm  : MPI communicator object
        """

        PETSc.Vec.__init__(self)
        self.createMPI(size, comm=comm)
        self.setUp()

        start = 0
        for r in xrange(0, comm.rank):
            start += size // comm.size + int(r < (size % comm.size))

        gindices = np.arange(start, start+len(data), dtype='int32')

        # Read in data
        self.setValues(gindices, data)

        self.assemblyBegin()
        self.assemblyEnd()
