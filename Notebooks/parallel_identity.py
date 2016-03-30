from badlands.petsc import coo_matrix
import numpy as np
import argparse
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser(description='Parse some model arguments.')
parser.add_argument('--n', type=int, required=True, help='Specify the matrix size')
args = parser.parse_args()

N = args.n # global size

displs = np.zeros(comm.size, dtype=int)
rcount = np.zeros(comm.size, dtype=int)

shadow = N // 10 # shadow region

start = 0
for r in xrange(0, comm.size):
    n = N // comm.size + int(r < (N % comm.size))
    rcount[r] = n + shadow
    displs[r] = start
    start += n

rcount[-1] = n # no shadow zone for last processor

print "Hi! I'm processor %i, I have rows %i to %i of %i" % (comm.rank, start, start+n, N)

n = rcount[comm.rank] # local size
start = displs[comm.rank] # offset

diag_IJ = np.arange(n, dtype='int32')
diag_V  = np.ones(n, dtype='float32')

t = time.clock()

A = coo_matrix(diag_IJ, diag_IJ, diag_V, shape=(N,N), offset=(start,n), comm=comm)
A.mat.view()
