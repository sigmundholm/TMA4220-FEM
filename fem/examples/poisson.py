import numpy as np
import matplotlib.pyplot as plt

from fem.fe_values import FE_Values
from fem.fe_q import FE_Q
from fem.function import Function
from fem.plotting import plot_mesh, plot_solution
from fem.supplied import getdisc
from fem.triangle import Cell
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

    def __init__(self, dim, degree, num_triangles, RHS):
        self.dim = dim
        self.degree = degree
        self.num_triangles = num_triangles

        self.RHS = RHS
        self.fe = FE_Q(dim, degree)

    def make_grid(self):
        print("Make grid")
        points, triangles, edges = getdisc.get_disk(self.num_triangles)
        # TODO med firkant som mesh funker ikke dirichlet på bdd.
        from fem.supplied import getplate
        # points, triangles, edges = getplate.get_plate(self.num_triangles)
        self.points = points
        self.triangles = triangles
        self.edges = edges
        plot_mesh(points, triangles, edges)
        plt.show()

    def setup_system(self):
        print("Setup system")
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
        print("Assemble system")
        guass = QGauss(dim=self.dim, degree=3)
        fe_values = FEValues(self.fe, guass, self.points, self.edges,
                              Poisson.is_dirichlet, update_gradients=True)

        rhs = self.RHS()
        boundary_values = BoundaryValues()

        for triangle in self.triangles:
            # Nå er vi i en celle
            # lar indeksen i punktlista være den globale nummereringen til
            # shape funksjoner
            # TODO build this initialization into the for-loop.
            cell = Cell(self.dim, triangle)
            fe_values.reinit(cell)

            for q_index in fe_values.quadrature_point_indices():
                x_q = fe_values.quadrature_point(q_index)

                for i in fe_values.dof_indices():
                    for j in fe_values.dof_indices():
                        res = fe_values.shape_grad(i, q_index) \
                              @ fe_values.shape_grad(j, q_index) \
                              * fe_values.JxW(q_index)  # (∇u_i, ∇v_j)

                        self.system_matrix[fe_values.local2global[i],
                                           fe_values.local2global[j]] += res

                    val = fe_values.shape_value(i, q_index) * rhs.value(x_q) \
                          * fe_values.JxW(q_index)  # (v_i, f)
                    self.system_rhs[fe_values.local2global[i]] += val

        # TODO this fixes so the matrix is invertible, but could just have
        # removed those dofs that are not a dof from the matrix, so this
        # wouldn't be needed. Would then have to set boundary values in a
        # different way after the system is solved.
        for i, point in enumerate(self.points):
            if fe_values.is_boundary[i] and Poisson.is_dirichlet(point):
                self.system_matrix[i, i] = 1
                # TODO this doesn't implement lifting functions, so this
                # only works with homogeneous dirichlet boundary conditions.
                self.system_rhs[i] = boundary_values.value(point)

    def set_boundary_conditions(self):
        pass

    def solve(self):
        print("Solve")
        self.solution = np.linalg.solve(self.system_matrix, self.system_rhs)

    def output_results(self):
        pass

    def run(self):
        self.make_grid()
        self.setup_system()
        self.assemble_system()
        self.solve()

        ax = plot_solution(self.points, self.solution, self.triangles)
        ax.set_title("numerical solution")
        plt.show()


if __name__ == '__main__':
    p = Poisson(2, 1, 200, RightHandSide)
    p.run()
