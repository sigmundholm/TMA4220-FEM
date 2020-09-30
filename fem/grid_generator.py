import numpy as np

from fem.supplied import getdisc
from fem.tria import Triangulation


class GridGenerator:

    @staticmethod
    def sphere(tria: Triangulation, n_triangles):
        if tria.dim == 1:
            tria.points = np.linspace(-1, 1, n_triangles + 1)
            tria.triangles = list(zip(range(len(tria.points)),
                                      range(1, len(tria.points))))
            # TODO

        if tria.dim == 2:
            points, triangles, edges = getdisc.get_disk(n_triangles)

            tria.points = points
            tria.triangles = triangles
            tria.edges = edges
