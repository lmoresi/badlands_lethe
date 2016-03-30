import numpy as np
import sys, petsc4py
from petsc4py import PETSc
from .matrix import spmatrix
from ._sparsetools import coo_tocsr, csr_tocoo, sum_duplicates


def coo_matrix(I, J, V, shape, offset=None, comm=None):
    """
    Initialises a sparse PETSc Matrix object from row, col, val input vectors
    Accepts COO matrix format vectors and converts them to CSR for PETSc.

        ARGUMENTS
            I, J, V    : row, col, val sparse matrix arrays (COO matrix format)
            shape      : global dimensions of the matrix -- tuple(nrows, ncols)
            ownership  : local offset and number of elements -- tuple(start, n)
            comm       : MPI communicator object -- use mpi4py.MPI.COMM_WORLD

        RETURNS
            PETSc_mat  : extended PETSc matrix object
    """
    # Initialise PETSc matrix object
    mat = PETSc.Mat()
    mat.create(comm=comm)
    mat.setType('aij')

    if offset != None:
        start, n = tuple(offset)
    else:
        # Here we assume data is read in by rows with no shadow regions
        n = shape[0] // comm.size + int(comm.rank < (shape[0] % comm.size))
        start = 0
        for r in xrange(0, comm.rank):
            start += shape[0] // comm.size + int(r < (shape[0] % comm.size))

    # Global and local sizes
    if comm.size > 1:
        mat.setSizes(((n, shape[0]), (n, shape[1])))
    else:
        mat.setSizes((n, n))

    gindices = np.arange(start, start+n, dtype='int32')

    # Allow matrix insertion using local indices [0:n+2]
    lgmap = PETSc.LGMap()
    lgmap.create(gindices, comm=comm)
    mat.setLGMap(lgmap, lgmap)

    ## COO matrix - row and column vectors are the same length
    I, J = I.astype('int32'), J.astype('int32')

    # Sum duplicate entries
    I, J, V = sum_duplicates(I, J, V)

    # Ensure vectors are in sequential order
    idx = np.lexsort([J, I])
    I, J, V = I[idx], J[idx], V[idx]

    # number of nonzeros
    nnz = np.bincount(I, minlength=n).astype('int32')

    # Convert to CSR format
    I = np.zeros(n+1, dtype='int32')
    I[1:nnz.size+1] = nnz
    indptr = np.cumsum(I).astype('int32')

    ## uncomment if you don't mind mallocs
    ## mallocs are expensive so it is best to avoid them if possible
    # mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    # Memory preallocation
    mat.setPreallocationNNZ(nnz)
    # mat.setPreallocationNNZ((nnz[gindices], nnz[~gindices]))

    # Read in data
    mat.setValuesLocalCSR(indptr, J, V)

    mat.assemblyBegin()
    mat.assemblyEnd()

    # spmatrix.__init__(self)
    return spmatrix(mat)
