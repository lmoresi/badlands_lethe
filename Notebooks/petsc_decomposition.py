'''
Creates a distributed unstructured mesh from a set of local points and triangulation.
'''
resI, resJ = 10, 10

from mpi4py import MPI
import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import badlands
import numpy as np

comm = MPI.COMM_WORLD


if not PETSc.COMM_WORLD.rank:
    Xcoords = np.linspace(0, 10, resI)
    Ycoords = np.linspace(-10, 0, resJ)

    xq, yq = np.meshgrid(Xcoords, Ycoords)
    x5, y5 = xq.flatten(), yq.flatten()

    tri = badlands.tools.Triangulation(x5, y5)
    tri.triangulate()
    print tri.npoints, tri.simplices.shape

    coords, cells = tri.points, tri.simplices

else:
    coords, cells = np.zeros((0,2), dtype=float), np.zeros((0,3), dtype=PETSc.IntType)



dim = 2
plex = PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=PETSc.COMM_WORLD)



pStart, pEnd = plex.getChart()
plex.view()
print "pStart, pEnd: ", pStart, pEnd



# Create section with 1 field with 1 DoF per vertex, edge amd cell
numComp = 1
# Start with an empty vector
numDof = [0] * 3
# Field defined on vertexes
numDof[0] = 1
# Field defined on edges
numDof[1] = 0
# Field defined on cells
numDof[2] = 0

origSect = plex.createSection(1, [1,0,0])
origSect.setFieldName(0, 'TestField')
origSect.setUp()
origSect.view()

plex.setDefaultSection(origSect)
origVec = plex.createGlobalVec()

# mat = plex.getMatrix()
# mat.view()
# print mat.getSizes()
# print comm.rank, mat.getLocalSize()

# Create origVec with global range
if not PETSc.COMM_WORLD.rank:
    origVec.setValues(range(pStart, pEnd), range(pStart, pEnd))


# In[37]:
# If size > 1 then distribute to other processors, otherwise keep as is!
if PETSc.COMM_WORLD.size > 1:
    sf = plex.distribute()
    sf.view()

    newSect, newVec = plex.distributeField(sf, origSect, origVec)

else:
    newSect = origSect
    newVec = origVec

newSect.view()
newVec.view()

# mat = plex.getMatrix()
# mat.view()
# print comm.rank, mat.getOwnershipRange()


v = plex.getCoordinatesLocal()
# coords = v.array.reshape(v.local_size/2, 2)
# x_local, y_local = coords[:,0], coords[:,1]

## Get all local coordinates on root
N = np.zeros(comm.size, dtype=int)
n = np.array(v.array.size, dtype=int)
comm.Allgather([n, MPI.INT], [N, MPI.INT]) # get sizes

local_v = np.asarray(v.array, dtype=float)
global_v = np.zeros(N.sum(), dtype=float)

rcount = N
displs = np.zeros(comm.size, dtype=int)

start = 0
for r in range(comm.size):
    displs[r] = start
    start += rcount[r]

comm.Gatherv([local_v, MPI.FLOAT], [global_v, rcount, displs, MPI.DOUBLE]) # global vector


if not PETSc.COMM_WORLD.rank:
    global_v = np.split(global_v, displs)[1:]

    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    colours = ['r','b','y','g','m','o']

    ax.triplot(coords[:,0], coords[:,1], cells, marker=None, linewidth=0.5, alpha=0.9/comm.size)
    for r in range(comm.size):
        local_coords = np.reshape(global_v[r], (rcount[r]/2, 2))
        local_x, local_y = local_coords[:,0], local_coords[:,1]
        ax.scatter(local_x, local_y, c=colours[r], s=50, alpha=1.0/comm.size)
    plt.show()
