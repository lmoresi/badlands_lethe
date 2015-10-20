"""
Module for generating and manipulating unstructured 2D meshes for surface process simulations

The module defines a meshing class and some tools to create simple standalone meshes and calculate derivatives etc

    TreMesh

The TreMesh class builds a Delaunay triangulation and some higher-level data structures for
consistent neighbourhood sweeping and differential operators (grad, div, del-squared) from
the triangulation.

Some plotting functions are included.

See the help for each class / function for more detailed information

"""

from .trimesh     import TriMesh
# from .meshshape   import Shape2Mesh
