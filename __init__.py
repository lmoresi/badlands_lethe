"""
Module for generating and manipulating unstructured 2D meshes for surface process simulations

The module defines three meshing classes and some tools to create simple standalone meshes

    SurfaceProcessMesh(HeightMesh)
    HeightMesh(TriMesh)
    TreMesh

The TreMesh class builds a Delaunay triangulation and some higher-level data structures for 
consistent neighbourhood sweeping and differential operators (grad, div, del-squared) from
the triangulation.

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

from .tremesh   import meshtools
from .tremesh   import TreMesh
from .tremesh   import Shape2Mesh
from .surfmesh  import HeightMesh
from .surfmesh  import SurfaceProcessMesh
from .thermesh  import ThermMesh

