import numpy as np
import sys, petsc4py
from petsc4py import PETSc
from .vector import Vector as _Vector
from ._sparsetools import coo_tocsr, csr_tocoo, sum_duplicates
petsc4py.init(sys.argv)

class spmatrix(object):
    """
    Subclass to extend PETSc matrix objects for easy matrix operations.
    Takes a PETSc matrix object, which can be created from:
        coo_matrix
        csr_matrix

    These methods are similar to those in the SciPy sparse matrix module,
    but for parallel operations.

    The PETSc Matrix object is stored in `self.mat` and can be referenced directly.
    However, the methods defined here should be sufficiently comprehensive.
    """

    name = "PETSc_Matrix"

    def __init__(self, PETSc_mat):
        """
        Accepts a PETSc matrix object

        Initialises the PETSc Matrix object to import methods for matrix multiplication, etc.
        Accepts COO matrix format vectors and converts them to CSR for PETSc.

            ARGUMENTS
                PETSc_Mat  : PETSc matrix object (created from coo_matrix or csr_matrix)
        """
        # Save PETSc matrix object
        self.mat = PETSc_mat

        # Save PETSc global variables
        self.start = self.mat.owner_range[0]
        self.sizes = self.mat.sizes
        self.shape = (self.sizes[0][1], self.sizes[1][1]) # global shape
        self.local_shape = (self.sizes[0][0], self.sizes[1][0]) # local shape
        # self.view()


    def copy(self):
        """ duplicate the matrix """
        return spmatrix(self.mat.duplicate())
    def todense(self):
        """ dense numpy matrix (for sequential matrices only) """
        return np.matrix(self.mat[:,:])
    def toarray(self):
        """ numpy array with the same dimensions (for sequential matrices only) """
        return np.array(self.mat[:,:])
    def diagonal(self):
        """ diagonal of the matrix """
        return self.mat.getDiagonal()[:]
    def transpose(self):
        """
        Transpose matrix
            C = A^T
        """
        return spmatrix(self.mat.transpose())

    def __mul__(self, B):
        """
        Matrix-Matrix or Matrix-Vector multiplication
            C = A * B
        """
        if isinstance(B, _Vector):
            vec = self.mat.createVecRight()
            self.mat.mult(B, vec)
            return vec.array
        elif isinstance(B, spmatrix):
            return spmatrix(self.mat.matMult(B.mat))
        elif isinstance(B, np.ndarray):
            B2, vec = self.mat.createVecs()
            gindices = np.arange(self.start, self.start+self.local_shape[0], dtype='int32')
            B2.setValues(gindices, B)
            self.mat.mult(B2, vec)
            return vec.array
        elif isinstance(B, (float, int, bool)):
            # element-wise multiplication
            indptr, indices, data = self.mat.getValuesCSR()
            M = spmatrix(self.mat.copy())
            gindices = np.arange(self.start, self.start+self.local_shape[0], dtype='int32')
            lgmap = PETSc.LGMap()
            lgmap.create(gindices, comm=self.mat.comm)
            M.mat.setLGMap(lgmap, lgmap)
            M.mat.setValuesLocalCSR(indptr, indices-self.start, data*float(B), addv=False)
            M.mat.assemble()
            return M
        else:
            raise NotImplementedError()

    def __imul__(self, B):
        """
        element-wise Matrix-Matrix multiplication
            A += B
        """
        if isinstance(B, spmatrix):
            mul_mat = self.mat.matMult(B.mat)
            self.__init__(mul_mat)
        else:
            raise TypeError('Not a valid PETSc object')

    def __rmul__(self, a):
        """
        element-wise scalar multiplication
            C = a * A
        """
        if isinstance(a, (float, int, bool)):
            # element-wise multiplication
            indptr, indices, data = self.mat.getValuesCSR()
            M = spmatrix(self.mat.copy())
            gindices = np.arange(self.start, self.start+self.local_shape[0], dtype='int32')
            lgmap = PETSc.LGMap()
            lgmap.create(gindices, comm=self.mat.comm)
            M.mat.setLGMap(lgmap, lgmap)
            M.mat.setValuesLocalCSR(indptr, indices-self.start, data*float(a), addv=False)
            M.mat.assemble()
            return M
        else:
            raise NotImplementedError()

    def dot(self, B):
        """
        Matrix-Matrix or Matrix-Vector multiplication
            C = A * B
        """
        return self.__mul__(B)

    def Tdot(self, B):
        """
        transposed Matrix-Matrix or Matrix-Vector multiplication
            C = A^T * B
        """
        if isinstance(B, _Vector):
            vec = self.mat.createVecRight()
            self.mat.multTranspose(B, vec)
            return vec.array
        elif isinstance(B, spmatrix):
            return spmatrix(self.mat.matTransposeMult(B.mat))
        elif isinstance(B, np.ndarray):
            B2, vec = self.mat.createVecs()
            gindices = np.arange(self.start, self.start+self.local_shape[0], dtype='int32')
            B2.setValues(gindices, B)
            self.mat.multTranspose(B2, vec)
            return vec.array
        elif isinstance(B, (float, int, bool)):
            indptr, indices, data = self.mat.getValuesCSR()
            M = spmatrix(self.mat.transpose())
            gindices = np.arange(self.start, self.start+self.local_shape[0], dtype='int32')
            lgmap = PETSc.LGMap()
            lgmap.create(gindices, comm=self.mat.comm)
            M.mat.setLGMap(lgmap, lgmap)
            M.mat.setValuesLocalCSR(indptr, indices-self.start, data*float(B), addv=False)
            M.mat.assemble()
            return M
        else:
            raise NotImplementedError()

    def __add__(self, B):
        """
        element-wise addition
            C = A + B
        """
        # print "addition time!"
        # Identify type and shape of B
        if isinstance(B, spmatrix):
            assert self.shape == B.shape
            A_indptr, A_indices, A_data = self.mat.getValuesCSR()
            B_indptr, B_indices, B_data = B.mat.getValuesCSR()
            # print self.mat.comm.rank, A_indices

            if (A_indptr == B_indptr).all() and (A_indices == B_indices).all():
                # Matrices have the same structure, fast addition is possible
                sum_mat = self.mat.copy()
                gindices = np.arange(self.start, self.start+self.local_shape[0], dtype='int32')
                lgmap = PETSc.LGMap()
                lgmap.create(gindices, comm=self.mat.comm)
                sum_mat.setLGMap(lgmap, lgmap)
                sum_mat.setValuesLocalCSR(B_indptr, B_indices-self.start, B_data, addv=True)
                sum_mat.assemble()
                return spmatrix(sum_mat)
            else:
                # Matrices have different structure
                sum_mat = PETSc.Mat()
                sum_mat.create(comm=self.mat.comm)
                sum_mat.setType('aij')
                sum_mat.setSizes(self.mat.sizes)
                gindices = np.arange(self.start, self.start+self.local_shape[0], dtype='int32')
                lgmap = PETSc.LGMap()
                lgmap.create(gindices, comm=self.mat.comm)
                sum_mat.setLGMap(lgmap, lgmap)
                nnz = np.diff(A_indptr) + np.diff(B_indptr)
                sum_mat.setPreallocationNNZ(nnz.astype('int32'))
                sum_mat.setValuesLocalCSR(A_indptr, A_indices-self.start, A_data, addv=True)
                sum_mat.setValuesLocalCSR(B_indptr, B_indices-self.start, B_data, addv=True)
                sum_mat.assemblyBegin()
                sum_mat.assemblyEnd()
                return spmatrix(sum_mat)

        elif isinstance(B, _Vector):
            raise NotImplementedError('Vector addition is not yet implemented')

        elif isinstance(B, (float, int, bool)):
            raise NotImplementedError('Scalar addition is not yet implemented')

        else:
            raise TypeError("Need a valid type:\n - PETSc Matrix\n - PETSc Vector\n - NumPy ndarray\n - float")

    def __iadd__(self, B):
        """
        element-wise in-place addition with another PETSc matrix
            A += B
        """
        # Identify type and shape of B
        if isinstance(B, spmatrix):
            assert self.shape == B.shape
            if (A_indptr == B_indptr).all() and (A_indices == B_indices).all():
                indptr, indices, data = B.mat.getValuesCSR()
                self.mat.setValuesCSR(indptr, indices, data, addv=True)
                self.mat.assemble()
            else:
                raise NotImplementedError()

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
        if isinstance(B, spmatrix):
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

        return spmatrix(sum_mat)

    def __isub__(self, B):
        """
        element-wise in-place subtraction with another PETSc matrix
            A += B
        """
        # Identify type and shape of B
        if isinstance(B, spmatrix):
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
