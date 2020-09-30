import numpy as np
import matplotlib.pyplot as plt

from fem.supplied import getdisc
from fem.plotting import plot_mesh
from fem.fe_values import FE_Values
from fem.fe_q import FE_Q
from fem.quadrature_lib import QGauss


class Poisson:
    num_triangles = 1

    points = None
    triangles = None
    edges = None

    n_dofs = None
    system_matrix = None
    system_rhs = None
    solution = None

    def __init__(self, dim, degree, num_triangles):
        self.dim = dim
        self.degree = degree
        self.num_triangles = num_triangles

        self.fe = FE_Q(dim, degree)

    def make_grid(self):
        points, triangles, edges = getdisc.get_disk(self.num_triangles)
        self.points = points
        self.triangles = triangles
        self.edges = edges
        plot_mesh(points, triangles, edges)
        plt.show()

    def setup_system(self):
        self.n_dofs = len(self.points)

        self.system_matrix = np.zeros((self.n_dofs, self.n_dofs))
        self.system_rhs = np.zeros((self.n_dofs, 1))
        self.solution = np.zeros((self.n_dofs, 1))

    def assemble_system(self):
        """
        Set up the stiffness matrix.
        :return:
        """
        quadrature = QGauss(dim=self.dim, degree=self.degree)
        fe_values = FE_Values(self.fe, quadrature, self.points,
                              update_gradients=True)

        for triangle in self.triangles:
            # Nå er vi i en celle
            # lar indeksen i punktlista være den globale nummereringen til
            # shape funksjoner

            # TODO pass på ikke ha shape funksjoner på de kantene langs
            # randen der det skal være dirichlet.

            fe_values.reinit(triangle)

            points = self.points[triangle]

            for i in fe_values.dof_indices():
                for j in fe_values.dof_indices():
                    # TODO gradient of linear shape functions are constant on
                    # each element.
                    expr = fe_values.shape_grad(i, None) \
                           @ fe_values.shape_grad(j, None)
                    # fe_values.shape_grad(j, None)
                    res = quadrature.quadrature(points, lambda x, y: expr)
                    self.system_matrix[i, j] = res

    def solve(self):
        pass

    def output_results(self):
        pass

    def run(self):
        self.make_grid()
        self.setup_system()
        self.assemble_system()


if __name__ == '__main__':
    p = Poisson(2, 1, 10)

    p.run()
