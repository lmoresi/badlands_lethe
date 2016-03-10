import triangle
from numpy import column_stack

class Triangulation(object):
    """
    An abstraction for the triangle python module <http://dzhelil.info/triangle/>
    This class makes triangulation routines more python-esc.

    Vanilla triangle requires a dictionary of vertices, segments, etc. followed
    by keyword arguments that toggle specific functionality.
    Here, keywords are defined explicitely and passed to triangle.
    """

    def __init__(self, points_x, points_y, segments=None, holes=None):
        """
        Initialise the data structures for the triangulation method

            ARGUMENTS
                points_x     : x points (flattened numpy array)
                points_y     : y points (flattened numpy array)

            OPTIONAL ARGUMENTS
                segments     : bounding box for the triangulation (integers)
                holes        : holes in the mesh (integers)
        """

        self.points = column_stack([points_x, points_y])
        self.npoints = len(self.points)
        self.segments = segments
        self.holes = holes

    def triangulate(self, q=None, a=None, p=False, D=False, c=False, **kwargs):
        """
        Create a Delaunay triangulation and store on the object

            ARGUMENTS
                q     (float) : quality mesh generation, no angles smaller than specified value
                a     (float) : imposes a maximum triangle area constraint
                p     (bool)  : triangulates a planar straight line graph (use with segments)
                D     (bool)  : switch to a conforming Delaunay mesh rather than constrained Delaunay
                c     (bool)  : encloses the convx hull with segments
                kwargs        : see documentation for more  <http://dzhelil.info/triangle/>
        """

        toggles = {'q':q, 'a':a, 'p':p, 'D':D, 'c':c}
        for key, value in kwargs.items():
            toggles[str(key)] = value

        # Construct options to pass to triangle

        options = ""
        for key, value in toggles.items():
            if value:
                if isinstance(value, bool):
                    options += str(key)
                elif isinstance(value, (float, int)):
                    options += str(key)+str(value)
                else:
                    raise TypeError('Argument is not a valid type')

        # Construct dictionary

        d = dict(vertices=self.points)
        if self.segments:
            d['segments'] = self.segments
        if self.holes:
            d['holes'] = self.holes

        if options:
            tri = triangle.triangulate(d, options)
        else:
            tri = triangle.triangulate(d)

        # Store triangulation
        self.simplices = tri['triangles']
