import numpy as np


class Face:
    # This is set as a static variable in FEValuesBase.__init__.
    # TODO do something less hacky
    vertex_at_boundary: dict = None

    # A list of all the points in the mesh. This is set as a static variable
    # in FEValuesBase.__init__.
    points = []

    def __init__(self, dim, edge_indices):
        self.dim = dim
        self.edge_indices = edge_indices

    def at_boundary(self):
        # If all the points of this face is on the boundary, this face is on
        # the boundary. In 1D, this will just test one point, but in 2D it
        # will test both endpoints of this face.
        return all(self.vertex_at_boundary[i] for i in self.edge_indices)


class Cell:
    # A list of all the points in the mesh. This is set as a static variable
    # in FEValuesBase.__init__.
    points = []

    def __init__(self, dim, corner_indices):
        self.dim = dim
        self.corner_indices = corner_indices
        self.corner_points = self.points[corner_indices]

    def face_iterators(self) -> [Face]:
        # Create list of tuples of all the pairs constituting the edges of
        # this cell (a triangle). In 1D a cell is just an interval, so then
        # the faces are just the two endpoints of this interval.
        if self.dim == 1:
            pairs = self.corner_indices
        elif self.dim == 2:
            pairs = list(zip(self.corner_indices, self.corner_indices[1:]))
            pairs.append((self.corner_indices[-1], self.corner_indices[0]))
        else:
            raise NotImplementedError()

        return [Face(self.dim, np.array(edge)) for edge in pairs]
