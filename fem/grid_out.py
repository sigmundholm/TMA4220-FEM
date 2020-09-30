import numpy as np
import matplotlib.pyplot as plt

from fem.tria import Triangulation


class GridOut:
    @staticmethod
    def write_svg(obj, out=None):
        """
        Plot the object if the output stream is None, else write it to the file.
        """
        if isinstance(obj, Triangulation):
            return GridOut.__write_svg_triangulation(tria, out)

    @staticmethod
    def __write_svg_triangulation(tria: Triangulation,
                                  out=None):
        print("hei")

        xs = tria.points[:, 0]
        ys = tria.points[:, 1]

        fig, ax = plt.subplots()

        ax.scatter(xs, ys)

        # Plot the edges of all triangles
        for triangle in tria.triangles:
            for edge in zip(triangle, list(triangle[1:]) + list([triangle[0]])):
                # TODO very slow..
                ax.plot(tria.points[edge, 0], tria.points[edge, 1],
                        color="black")

        return ax

    def write_vtk(self, obj):
        """
        Write the object to a vtk file.
        :param obj:
        :return:
        """
        pass


if __name__ == '__main__':
    from fem.grid_generator import GridGenerator

    # Create grid
    tria = Triangulation(dim=2)
    GridGenerator.sphere(tria, n_triangles=30)

    GridOut.write_svg(tria)
    plt.show()
