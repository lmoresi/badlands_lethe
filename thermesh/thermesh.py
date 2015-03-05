## Temperature mesh (subclass of mesh) - this defines a TriMesh plus temperature and various functions
    

import numpy as np
import math
from .. import TreMesh


class ThermMesh(TreMesh):
    """
    Builds a TreMesh (2D) object and adds a temperature field and data structures / operators
    to propagate information across the surface (e.g. flow downhill) 
    """
    
    name="Generic_Height_TreMesh"

    def __init__(self, points_x, points_y, temperature, diffusivity, heat_sources, boundary_mask, verbose=False):
        """
        Initialise the Delaunay mesh (parent) and build height data structures
        """
        
        # initialise the mesh itself from the parent TreMesh class
        TreMesh.__init__(self, points_x, points_y, boundary_mask, verbose=verbose)

        
        # Add the temperature field 
        self.temperature = temperature
        self.diffusivity = diffusivity
        self.heat_sources = heat_sources
            
        return

 
 




