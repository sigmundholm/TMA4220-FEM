import numpy as np
import matplotlib.pyplot as plt

from fem.fe_values import FE_Values
from fem.fe_q import FE_Q
from fem.function import Function
from fem.plotting import plot_mesh
from fem.supplied import getdisc
from fem.quadrature_lib import QGauss


class RightHandSide(Function):
    def value(self, p):
        return 1


class BoundaryValues(Function):
    def value(self, p):
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

    @staticmethod
    def is_dirichlet(p: np.ndarray):
        return True

    def assemble_system(self):
        """
        Set up the stiffness matrix.
        :return:
        """
        guass = QGauss(dim=self.dim, degree=self.degree)
        fe_values = FE_Values(self.fe, guass, self.points, self.edges,
                              Poisson.is_dirichlet, update_gradients=True)

        rhs = RightHandSide()
        boundary_values = BoundaryValues()

        for triangle in self.triangles:
            # Nå er vi i en celle
            # lar indeksen i punktlista være den globale nummereringen til
            # shape funksjoner

            # TODO pass på ikke ha shape funksjoner på de kantene langs
            # randen der det skal være dirichlet.

            fe_values.reinit(triangle)

            for q_index in fe_values.quadrature_point_indices():
                print("q index", q_index)
                x_q = fe_values.quadrature_point(q_index)

                for i in fe_values.dof_indices():
                    print()
                    for j in fe_values.dof_indices():
                        res = fe_values.shape_grad(i, q_index) \
                              @ fe_values.shape_grad(j, q_index) \
                              * fe_values.JxW(q_index)

                        self.system_matrix[fe_values.local2global[i],
                                           fe_values.local2global[j]] += res

                    # TODO pass på det dette integralet er positivt for
                    # poisson med f=1.
                    val = fe_values.shape_value(i, q_index) * rhs.value(x_q) \
                          * fe_values.JxW(q_index)
                    print("integral", val)
                    shape_val = fe_values.shape_value(i, q_index)
                    if shape_val < 0 or shape_val > 1:
                        print("shape-val", shape_val)
                    else:
                        print("ok")
                    self.system_rhs[fe_values.local2global[i]] += val

        # print(self.points)
        print(self.system_matrix)
        print(self.system_rhs)
        # TODO this fixes so the matrix is invertible, but could just have
        # removed those dofs that are not a dof from the matrix, so this
        # wouldn't be needed. Would then have to set boundary values in a
        # different way after the system is solved.

        for i, point in enumerate(self.points):
            if fe_values.is_boundary[i] and Poisson.is_dirichlet(point):
                self.system_matrix[i, i] = 1
                self.system_rhs[i] = boundary_values.value(point)

    def set_boundary_conditions(self):
        pass

    def solve(self):
        self.solution = np.linalg.solve(self.system_matrix, self.system_rhs)

    def output_results(self):
        pass

    def run(self):
        self.make_grid()
        self.setup_system()
        self.assemble_system()
        self.solve()

        plt.imshow(self.system_matrix)
        plt.show()
        print("solution")
        print(self.solution)


if __name__ == '__main__':
    p = Poisson(2, 1, 20)
    p.run()
