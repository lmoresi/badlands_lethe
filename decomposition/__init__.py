"""
Module for decomposing global meshes into smaller portions to hand off to multiple processors

The module defines a decomposition class with various methods to decompose a global mesh

    MeshDecomposition

MeshDecomposition methods should be called in place of the build_mesh() method
of TriMesh or PixMesh classes if more than one processor should be used.
Problems on a single processor need not use these methods.

See the help for each class / function for more detailed information

"""

from .decomposition import MeshDecomposition
