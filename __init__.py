"""
Module for generating and manipulating unstructured 2D meshes for surface process simulations

The module defines four meshing classes and some tools to create simple standalone meshes

    SurfaceProcessMesh(HeightMesh)
    HeightMesh(TriMesh | PixMesh)
    TriMesh(MeshDecomposition) | PixMesh(MeshDecomposition)
    MeshDecomposition

The MeshDecomposition class divides a global mesh into local portions to be distributed
among processors.

The TriMesh class builds a Delaunay triangulation and some higher-level data structures for
consistent neighbourhood sweeping and differential operators (grad, div, del-squared) from
the triangulation.

The PixMesh class creates neighbourhood arrays and differential operators for a
quadrilateral mesh and mimics many of the data structures of the TriMesh class.

The HeightMesh class adds a height variable and operations to walk downhill
(or uphill by inverting the data !). It gives the capacity to step information downhill node-to-node
using the scipy.sparse matrix classes. It also provides the capacity to flow information to the boundaries
or integrate information to the boundaries.

The SurfaceProcessMesh class adds the specifics needed to interpret a HeightMesh as topography
subject to simple erosion / transport models. This includes calculating catchments, aggregating rainfall,
computing erosion rates / deposition rates.

Some plotting functions are included.

See the help for each class / function for more detailed information

"""


import tools as tools

import petsc

from .decomposition  import MeshDecomposition as _MeshDecomposition
from .trimesh        import TriMesh  as _TriMesh
from .pixmesh        import PixMesh  as _PixMesh
from .topomesh       import TopoMesh as _TopoMeshClass
from .surfmesh       import SurfaceProcessMesh as _SurfaceProcessMeshClass

known_basemesh_classes = { "TriMesh" : _TriMesh,
                           "PixMesh" : _PixMesh }

## These are factory functions for the TopoMesh class and the
## SurfaceProcessMesh class. They bundle the chosen mesh class in
## underneath the topography and surface process functions.
## For consistency of the interface, we define a FlatMesh factory
## function here as well even though the TriMesh and PixMesh are
## equivalent

def FlatMesh(BaseMeshType):

    if BaseMeshType in known_basemesh_classes.keys():
        class FlatMeshClass(known_basemesh_classes[BaseMeshType], _MeshDecomposition):
            pass

        return FlatMeshClass()

    else:
        print "Warning !! Mesh type {:s} unknown".format(BaseMeshType)
        print "Known mesh types: {}".format(known_mesh_classes.keys())

    return

def TopoMesh(BaseMeshType):

    if BaseMeshType in known_basemesh_classes.keys():
        class TopoMeshClass(_TopoMeshClass, known_basemesh_classes[BaseMeshType], _MeshDecomposition):
            pass

        return TopoMeshClass()

    else:
        print "Warning !! Mesh type {:s} unknown".format(BaseMeshType)
        print "Known mesh types: {}".format(known_mesh_classes.keys())


    return

def SurfaceProcessMesh(BaseMeshType):

    if BaseMeshType in known_basemesh_classes.keys():
        class SurfaceProcessMeshClass(_SurfaceProcessMeshClass, _TopoMeshClass, known_basemesh_classes[BaseMeshType], _MeshDecomposition):
            pass

        return SurfaceProcessMeshClass()

    else:
        print "Warning !! Mesh type {:s} unknown".format(BaseMeshType)
        print "Known mesh types: {}".format(known_mesh_classes.keys())

    return
