import time
import numpy as np
import argparse
from mpi4py import MPI
from badlands import petsc as solvers
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser(description='Parse some model arguments.')
parser.add_argument('--res', type=int, nargs=2, required=True, metavar=('X', 'Y'), help='Resolution in X and Y directions')
parser.add_argument('--plot', action='store_true', required=False, default=False, help='Save a plot')
parser.add_argument('--view', action='store_true', required=False, default=False, help='Show matrix and vector properties in the terminal')
args = parser.parse_args()

class HotBox:
    """
    Simple conduction example with Dirichlet boundary conditions
    """
    def __init__(self, nx, ny, kappa, H):

        self.kappa = np.asarray(kappa, dtype='f')
        self.H = np.asarray(H, dtype='f')

        nodes = np.arange(nx*ny).reshape(ny,nx)
        n = nx*ny
        self.nx, self.ny = nx, ny
        self.dx, self.dy = 1., 1.

        # boundary nodes
        self.top_nodes = nodes[0,:]
        self.bottom_nodes = nodes[-1,:]
        self.left_nodes = nodes[1:-1,0]
        self.right_nodes = nodes[1:-1,-1]

        # Mask boundary
        bmask = np.zeros_like(nodes, dtype=bool)
        bmask[1:-1,1:-1] = True
        self.nodes = nodes[bmask].flatten()

        # Sparse IJV vectors
        self.I = np.zeros(n*5, dtype=int)
        self.J = np.zeros(n*5, dtype=int)
        self.V = np.zeros(n*5)

        # RHS matrix
        self.RHS = np.zeros(n)

    def sle(self, bc=[0., 1., 0., 1.]):
        """
        Return a system of linear equation and a RHS vector
        """
        self.nPos = 0

        # Boundaries
        for i, idx in enumerate([self.top_nodes, self.right_nodes, self.bottom_nodes, self.left_nodes]):
            for index in idx:
                self.write_matrix(index, index, 1.0)
                self.RHS[index] = bc[i]

        # Main finite differences setup
        for idx in self.nodes:
            self.write_matrix(idx, idx-self.nx, self.kappa/self.dy**2)
            self.write_matrix(idx, idx-1, self.kappa/self.dx**2)
            self.write_matrix(idx, idx, -2.*self.kappa/self.dx**2 - 2.*self.kappa/self.dy**2)
            self.write_matrix(idx, idx+1, self.kappa/self.dx**2)
            self.write_matrix(idx, idx+self.nx, self.kappa/self.dy**2)

            self.RHS[idx] = -self.H

        return self.I, self.J, self.V, self.RHS

    def write_matrix(self, I, J, V):
        self.I[self.nPos] = I
        self.J[self.nPos] = J
        self.V[self.nPos] = V
        self.nPos += 1


# Set up aij vectors for sparse matrix operations
nx, ny = args.res[0], args.res[1]
N = nx*ny

box = HotBox(nx, ny, 1.0, 2e-6)
I, J, V, rhs = box.sle()

## Divide up between processors
mask = V != 0
I, J, V = I[mask], J[mask], V[mask] # remove unwanted entries

# Sort I,J,V vectors
idx = np.lexsort([J, I])
I, J, V = I[idx], J[idx], V[idx]


displs = np.zeros(comm.size, dtype=int)
rcount = np.zeros(comm.size, dtype=int)

start = 0
for r in xrange(0, comm.size):
    rcount[r] = N // comm.size + int(r < (N % comm.size))
    displs[r] = start
    start += rcount[r]

n = rcount[comm.rank]
start = displs[comm.rank]

print "Hi! I'm processor %i, I have rows %i to %i of %i" % (comm.rank, start, start+n, N)

local_mask = np.logical_and(I >= start, I < start+n)

local_I = I[local_mask]
local_J = J[local_mask]
local_V = V[local_mask]
local_rhs = rhs[start:start+n]


## Give to PETSc Matrix class
t = time.clock()
A = solvers.Matrix(local_I, local_J, local_V, shape=(N,N), comm=comm)
b = solvers.Vector(local_rhs, rhs.size, comm=comm)


if args.view:
    print "PETSc matrix properties"
    A.mat.view()
    print "PETSc vector properties"
    b.view()

if comm.rank == 0:
    print "PETSc matrix and vector construction: {} s".format(time.clock()-t)


row_format = " {0:10} {1:.4f} seconds"
if comm.rank == 0:
    print "\nTimings for arithmetic operations using a system of linear equations: Ax = b"
    print "---------------------------"
    print "Operation \t time"
    print "---------------------------"

t = time.clock()
A2 = A * A
if comm.rank == 0:
    print row_format.format('A * A', time.clock()-t)


t = time.clock()
x = A * b
if comm.rank == 0:
    print row_format.format('A * b', time.clock()-t)


t = time.clock()
A2 = A + A
if comm.rank == 0:
    print row_format.format('A + A', time.clock()-t)


# Solve Ax=b for x
t = time.clock()
T = solvers.solve(A, b)
if comm.rank == 0:
    print row_format.format('A^-1 b', time.clock()-t)
    print "---------------------------"

# Gather the solution to one array
T_gather = np.zeros(N)
comm.Allgatherv([T, MPI.FLOAT], [T_gather, rcount, displs, MPI.FLOAT_INT])

if comm.rank == 0 and args.plot:
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    im = ax.imshow(T_gather.reshape(ny,nx), interpolation='none')
    fig.colorbar(im)
    plt.show()
