import numpy as np
import sys, petsc4py
from petsc4py import PETSc
from .matrix import spmatrix
from ._sparsetools import coo_tocsr, csr_tocoo, sum_duplicates


def csr_matrix(indptr, indices, data, shape, offset=None, comm=None):
    """
    Initialises a sparse PETSc Matrix object from indptr, indices, data input vectors
    Accepts CSR matrix format for fast PETSc matrix creation

        ARGUMENTS
            indptr     : points to row breaks in indices (length is number of rows in matrix)
            indices    : column indices (same length as data)
            data       : data (same length as indices)
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

    indptr = indptr.astype('int32')
    indices = indices.astype('int32')
    data = data.astype('int32')

    # number of nonzeros
    nnz = indptr[1:].astype('int32')

    ## uncomment if you don't mind mallocs
    ## mallocs are expensive so it is best to avoid them if possible
    # mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    # Memory preallocation
    mat.setPreallocationNNZ(nnz)
    # mat.setPreallocationNNZ((nnz[gindices], nnz[~gindices]))

    # Read in data
    mat.setValuesLocalCSR(indptr, indices, data)

    mat.assemblyBegin()
    mat.assemblyEnd()

    return spmatrix(mat)
