import numpy as np
import sys, petsc4py
from petsc4py import PETSc
from .vector import Vector as _Vector
from ._sparsetools import coo_tocsr, csr_tocoo, sum_duplicates
petsc4py.init(sys.argv)

class Matrix(PETSc.Mat):
    """
    Extends the PETSc.Mat class with additional methods and tools for easy matrix operations.

    All matrix operations must be carried out with PETSc objects defined here.
    """

    name = "PETSc_Matrix"

    def __init__(self, I, J, V, shape, comm):
        """
        Initialises the PETSc Matrix object to import methods for matrix multiplication, etc.
        Accepts COO matrix format vectors and converts them to CSR for PETSc.

            ARGUMENTS
                I, J, V  : row, col, val sparse arrays arrays (COO matrix format)
                shape    : global dimensions of the matrix -- tuple(nrows, ncols)
                comm     : MPI communicator object
        """

        # Convert integers to dtype int32
        I = I.astype('int32')
        J = J.astype('int32')

        self.shape = shape

        # Sum duplicate entries
        I, J, V = sum_duplicates(I, J, V)

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
        PETSc.Mat.__init__(self)
        self.create(comm=comm)
        self.setType('aij')

        if comm.size > 1:
            self.setSizes(((n, shape[0]), (n, shape[1])))
        else:
            self.setSizes((n, n))

        gindices = np.arange(start, start+n, dtype='int32')

        # self.setPreallocationNNZ((nnz[gindices], nnz[~gindices]))
        self.setPreallocationNNZ(nnz)

        # Allow matrix insertion using local indices [0:n+2]
        lgmap = PETSc.LGMap()
        lgmap.create(list(gindices), comm=comm)
        self.setLGMap(lgmap, lgmap)

        # Read in data
        self.setValuesCSR(np.insert(np.cumsum(nnz),0,0).astype('int32'), J, V)

        # depreciated method to read in data
        # for i, row in enumerate(gindices):
        #     self[row, cols[i]] = data[i]

        self.start = start
        self.assemblyBegin()
        self.assemblyEnd()

    def todense(self):
        """ dense numpy matrix """
        return np.matrix(self[:,:])
    def toarray(self):
        """ numpy array with the same dimensions """
        return np.array(self[:,:])
    def diagonal(self):
        """ diagonal of the matrix """
        return self.getDiagonal()[:]

    def __mul__(self, B):
        """
        Matrix-Matrix or Matrix-Vector multiplication
            C = A * B
        """
        if isinstance(B, _Vector):
            vec = self.createVecRight()
            self.mult(B, vec)
            return vec
        elif isinstance(B, Matrix):
            return self.matMult(B)
        else:
            raise NotImplementedError()

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
        shape = self.getSize()
        indptr, indices, data = self.getValuesCSR()
        # A_idxc = np.split(indices, indptr)
        # A_data = np.split(data, indptr)

        I, J, V = csr_tocoo(indptr, indices, data)

        # Identify type and shape of B
        if isinstance(B, Matrix):
            assert self.shape == B.shape
            B_indptr, B_indices, B_data = B.getValuesCSR()
            BI, BJ, BV = csr_tocoo(B_indptr, B_indices, B_data)
            I, J, V = np.append(I, BI), np.append(J, BJ), np.append(V, BV)
            # I, J, V = sum_duplicates(I, J, V)

            # for row, cols in enumerate(A_indxc[1:-1]):

        # elif isinstance(B, (np.array(0), list)):
        #     B = np.asarray(B)
        #     assert B.size in shape

        # elif isinstance(B, (float, int, bool)):

        else:
            raise TypeError("Need a valid type:\n - PETSc Matrix\n - NumPy ndarray\n - float")

        return Matrix(self.start+I, J, V, self.shape, self.comm)

    def __iadd__(self, B):
        """
        element-wise in-place addition with another PETSc matrix
            A += B
        """
        shape = self.getSize()
        I, J, V = B.getValuesCSR()

        # Identify type and shape of B
        if isinstance(B, Matrix):
            IB, JB, VB = B.getValuesCSR()
            I, J, V = np.append(I, IB), np.append(J, JB), np.append(V, VB)

        # elif isinstance(B, (np.array(0), list)):
        #     B = np.asarray(B)
        #     assert B.size in shape

        # elif isinstance(B, (float, int, bool)):

        else:
            raise TypeError("Need a valid type:\n - PETSc Matrix\n - NumPy ndarray\n - float")

        self.__init__(I, J, V, shape, self.comm)
