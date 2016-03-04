"""
These tools are useful for manipulating sparse matrix operations

"""

import numpy as np


def csr_tocoo(indptr, indices, data):
    """ Convert from CSR to COO sparse matrix format """
    I = np.zeros_like(indices)
    I[indptr[1:-1]] += 1
    return np.cumsum(I, dtype='int32'), indices, data

def coo_tocsr(I, J, V):
    """ Convert from COO to CSR sparse matrix format """
    nnz = np.bincount(I)[start:start+n].astype('int32')
    indptr = np.insert(np.cumsum(nnz),0,0).astype('int32')
    return indptr, J, V

def sum_duplicates(I, J, V):
    """
    Sum all duplicate entries in the matrix
    """
    order = np.lexsort((I, J))
    I, J, V = I[order], J[order], V[order]
    unique_mask = ((I[1:] != I[:-1]) |
                   (J[1:] != J[:-1]))
    unique_mask = np.append(True, unique_mask)
    unique_inds, = np.nonzero(unique_mask)
    return I[unique_mask], J[unique_mask], np.add.reduceat(V, unique_inds)
