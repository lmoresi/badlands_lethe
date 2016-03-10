import numpy as np
import sys, petsc4py
from petsc4py import PETSc
from .vector import Vector as _Vector
from ._sparsetools import coo_tocsr, csr_tocoo, sum_duplicates
petsc4py.init(sys.argv)

class Matrix(object):
    """
    Extends the PETSc.Mat class with additional methods and tools for easy matrix operations.
    These methods are similar to those in the SciPy sparse matrix module, but for parallel operations.

    The PETSc Matrix object is stored in `self.mat` and can be referenced directly.
    However, the methods defined here should be sufficiently comprehensive.
    """

    name = "PETSc_Matrix"

    def __init__(self, I=None, J=None, V=None, shape=None, comm=None, PETSc_mat=None):
        """
        Initialises the PETSc Matrix object to import methods for matrix multiplication, etc.
        Accepts COO matrix format vectors and converts them to CSR for PETSc.

            ARGUMENTS
                I, J, V    : row, col, val sparse matrix arrays (COO matrix format)
                shape      : global dimensions of the matrix -- tuple(nrows, ncols)
                comm       : MPI communicator object -- use mpi4py.MPI.COMM_WORLD

                PETSc_Mat  : existing matrix object (other arguments are not required)
        """

        if PETSc_mat:
            self.mat = PETSc_mat
        else:
            # Sum duplicate entries
            I, J, V = sum_duplicates(I.astype('int32'), J.astype('int32'), V)

            ## Here we assume data is read in by rows
            n = shape[0] // comm.size + int(comm.rank < (shape[0] % comm.size))
            start = 0
            for r in xrange(0, comm.rank):
                start += shape[0] // comm.size + int(r < (shape[0] % comm.size))

            nnz = np.bincount(I)[start:start+n].astype('int32')

            # Ensure vectors are in sequential order
            idx = np.lexsort([J, I])
            I, J, V = I[idx], J[idx], V[idx]

            # Initialise PETSc matrix object
            self.mat = PETSc.Mat()
            self.mat.create(comm=comm)
            self.mat.setType('aij')

            if comm.size > 1:
                self.mat.setSizes(((n, shape[0]), (n, shape[1])))
            else:
                self.mat.setSizes((n, n))

            gindices = np.arange(start, start+n, dtype='int32')

            # self.mat.setPreallocationNNZ((nnz[gindices], nnz[~gindices]))
            self.mat.setPreallocationNNZ(nnz)

            # Allow matrix insertion using local indices [0:n+2]
            lgmap = PETSc.LGMap()
            lgmap.create(list(gindices), comm=comm)
            self.mat.setLGMap(lgmap, lgmap)

            # Read in data
            self.mat.setValuesCSR(np.insert(np.cumsum(nnz),0,0).astype('int32'), J, V)

            self.mat.assemblyBegin()
            self.mat.assemblyEnd()


        # Query PETSc properties
        self.start = self.mat.owner_range[0]
        self.sizes = self.mat.sizes
        self.shape = (self.sizes[0][1], self.sizes[1][1]) # global shape
        self.local_shape = (self.sizes[0][0], self.sizes[1][0]) # local shape



    def todense(self):
        """ dense numpy matrix (for sequential matrices only) """
        return np.matrix(self.mat[:,:])
    def toarray(self):
        """ numpy array with the same dimensions (for sequential matrices only) """
        return np.array(self.mat[:,:])
    def diagonal(self):
        """ diagonal of the matrix """
        return self.mat.getDiagonal()[:]

    def __mul__(self, B):
        """
        Matrix-Matrix or Matrix-Vector multiplication
            C = A * B
        """
        if isinstance(B, _Vector):
            vec = self.mat.createVecRight()
            self.mat.mult(B, vec)
            return vec
        elif isinstance(B, Matrix):
            return Matrix(PETSc_mat=self.mat.matMult(B.mat))
        else:
            raise NotImplementedError()

    def __imul__(self, B):
        """
        element-wise Matrix-Matrix multiplication
        """
        if isinstance(B, Matrix):
            mul_mat = self.mat.matMult(B.mat)
            self.__init__(PETSc_mat=mul_mat)
        else:
            raise TypeError('Not a valid PETSc object')

    def dot(self, B):
        """
        Matrix-Matrix or Matrix-Vector multiplication
            C = A * B
        """
        self.__mul__(B)

    def __add__(self, B):
        """
        element-wise addition
            C = A + B
        """
        sum_mat = self.mat.copy()

        # Identify type and shape of B
        if isinstance(B, Matrix):
            assert self.shape == B.shape
            indptr, indices, data = B.mat.getValuesCSR()
            sum_mat.setValuesCSR(indptr, indices, data, addv=True)
            sum_mat.assemble()

        elif isinstance(B, _Vector):
            raise NotImplementedError()

        elif isinstance(B, (float, int, bool)):
            raise NotImplementedError()

        else:
            raise TypeError("Need a valid type:\n - PETSc Matrix\n - PETSc Vector\n - NumPy ndarray\n - float")

        return Matrix(PETSc_mat=sum_mat)

    def __iadd__(self, B):
        """
        element-wise in-place addition with another PETSc matrix
            A += B
        """
        # Identify type and shape of B
        if isinstance(B, Matrix):
            assert self.shape == B.shape
            indptr, indices, data = B.mat.getValuesCSR()
            self.mat.setValuesCSR(indptr, indices, data, addv=True)
            self.mat.assemble()

        elif isinstance(B, _Vector):
            raise NotImplementedError()

        elif isinstance(B, (float, int, bool)):
            raise NotImplementedError()

        else:
            raise TypeError("Need a valid type:\n - PETSc Matrix\n - PETSc Vector\n - NumPy ndarray\n - float")

    def __sub__(self, B):
        """
        element-wise subtraction
            C = A - B
        """
        sum_mat = self.mat.copy()

        # Identify type and shape of B
        if isinstance(B, Matrix):
            assert self.shape == B.shape
            indptr, indices, data = B.mat.getValuesCSR()
            sum_mat.setValuesCSR(indptr, indices, -data, addv=True)
            sum_mat.assemble()

        elif isinstance(B, _Vector):
            raise NotImplementedError()

        elif isinstance(B, (float, int, bool)):
            raise NotImplementedError()

        else:
            raise TypeError("Need a valid type:\n - PETSc Matrix\n - PETSc Vector\n - NumPy ndarray\n - float")

        return Matrix(PETSc_mat=sum_mat)

    def __isub__(self, B):
        """
        element-wise in-place subtraction with another PETSc matrix
            A += B
        """
        # Identify type and shape of B
        if isinstance(B, Matrix):
            assert self.shape == B.shape
            indptr, indices, data = B.mat.getValuesCSR()
            self.mat.setValuesCSR(indptr, indices, -data, addv=True)
            self.mat.assemble()

        elif isinstance(B, _Vector):
            raise NotImplementedError()

        elif isinstance(B, (float, int, bool)):
            raise NotImplementedError()

        else:
            raise TypeError("Need a valid type:\n - PETSc Matrix\n - PETSc Vector\n - NumPy ndarray\n - float")
