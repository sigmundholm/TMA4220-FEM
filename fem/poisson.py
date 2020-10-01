import numpy as np
import matplotlib.pyplot as plt

from fem.fe_values import FE_Values
from fem.fe_q import FE_Q
from fem.function import Function
from fem.plotting import plot_mesh
from fem.supplied import getdisc
from fem.quadrature_lib import QGauss


class RightHandSide(Function):
    def value(self, x, y):
        return 1


class BoundaryValues(Function):
    def value(self, x, y):
        return 0


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
        guass = QGauss(dim=self.dim, degree=self.degree)
        fe_values = FE_Values(self.fe, guass, self.points, self.edges,
                              update_gradients=True)
        rhs = RightHandSide()

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
                    res = guass.quadrature(points, lambda x, y: expr)
                    self.system_matrix[fe_values.local2global[i],
                                       fe_values.local2global[j]] = res

                def rhs_integrand(x, y):
                    return fe_values.shape_value(i, x, y) \
                           * rhs.value(x, y)

                val = guass.quadrature(points, rhs_integrand)
                self.system_rhs[fe_values.local2global[i]] = val

        print(self.points)
        print(self.system_matrix)
        print(self.system_rhs)

    def set_boundary_conditions(self):
        pass

    def solve(self):
        pass

    def output_results(self):
        pass

    def run(self):
        self.make_grid()
        self.setup_system()
        self.assemble_system()

        plt.imshow(self.system_matrix)
        plt.show()


if __name__ == '__main__':
    p = Poisson(2, 1, 20)

    p.run()
