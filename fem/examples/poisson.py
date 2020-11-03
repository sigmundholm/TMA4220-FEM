import numpy as np
import matplotlib.pyplot as plt

from fem.fe.fe_values import FEFaceValues, FEValues
from fem.fe.fe_q import FE_Q
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


class NeumannBoundaryValues(Function):
    def value(self, p):
        return 0


class Poisson:
    num_triangles = 1
    h = 0  # Length of longest edge in a triangle in the mesh.

    points = None
    triangles = None
    edges = None

    n_dofs = None
    system_matrix = None
    system_rhs = None
    solution = None

    def __init__(self, dim, degree, num_triangles, RHS, NeumannBD,
                 is_dirichlet: callable, quad_degree):
        self.dim = dim
        self.degree = degree
        self.num_triangles = num_triangles

        self.RHS = RHS
        self.NeumannBD = NeumannBD
        self.fe = FE_Q(dim, degree)

        # A function taking a point (np.ndarray) as argument, and returning
        # True if we should have Dirichlet boundary conditions here,
        # and False else.
        self.is_dirichlet = is_dirichlet

        self.quad_degree = quad_degree

    def make_grid(self):
        print("Make grid")
        points, triangles, edges = getdisc.get_disk(self.num_triangles,
                                                    dim=self.dim)
        # TODO med firkant som mesh funker ikke dirichlet på bdd.
        # points, triangles, edges = getplate.get_plate(self.num_triangles)
        self.points = points
        self.triangles = triangles
        self.edges = edges
        plot_mesh(points, triangles, edges)

        h = 0
        Cell.points = self.points

        for triangle in self.triangles:
            p1, p2, p3 = self.points[triangle]
            for p1, p2 in [(p1, p2), (p2, p3), (p3, p1)]:
                h_k = np.sqrt(((p2 - p1) ** 2).sum())
                if h_k > h:
                    h = h_k

        self.h = h
        print("   h =", h)

    def setup_system(self):
        print("Setup system")
        self.n_dofs = len(self.points)

        self.system_matrix = np.zeros((self.n_dofs, self.n_dofs))
        self.system_rhs = np.zeros((self.n_dofs, 1))
        self.solution = np.zeros((self.n_dofs, 1))

    def assemble_system(self):
        """
        Set up the stiffness matrix.
        :return:
        """
        print("Assemble system")
        gauss = QGauss(dim=self.dim, n=self.quad_degree)
        fe_values = FEValues(self.fe, gauss, self.points, self.edges,
                             self.is_dirichlet, update_gradients=True)

        face_gauss = QGauss(dim=self.dim - 1, n=self.quad_degree)
        fe_face_values = FEFaceValues(self.fe, face_gauss, self.points,
                                      self.edges, self.is_dirichlet)

        rhs = self.RHS()
        # TODO only homogeneous Dirichlet supperted
        boundary_values = BoundaryValues()
        neumann_bdd_values = self.NeumannBD()

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
                        val = fe_values.shape_grad(i, q_index) \
                              @ fe_values.shape_grad(j, q_index) \
                              * fe_values.JxW(q_index)  # (∇u_i, ∇v_j)

                        self.system_matrix[fe_values.local2global[i],
                                           fe_values.local2global[j]] += val

                    val = fe_values.shape_value(i, q_index) * rhs.value(x_q) \
                          * fe_values.JxW(q_index)  # (v_i, f)
                    self.system_rhs[fe_values.local2global[i]] += val

            for face in cell.face_iterators():
                if not face.at_boundary():
                    continue

                fe_face_values.reinit(cell, face)

                for q_index in fe_face_values.quadrature_point_indices():
                    x_q = fe_face_values.quadrature_point(q_index)

                    for j in fe_face_values.dof_indices():
                        g = neumann_bdd_values.value(x_q)
                        val = fe_face_values.shape_value(j, q_index) * g \
                              * fe_face_values.JxW(q_index)  # (g, v_j)
                        self.system_rhs[fe_face_values.local2global[j]] += val

        # TODO this fixes so the matrix is invertible, but could just have
        # removed those dofs that are not a dof from the matrix, so this
        # wouldn't be needed. Would then have to set boundary values in a
        # different way after the system is solved.
        for i, point in enumerate(self.points):
            if fe_values.is_boundary[i] and self.is_dirichlet(point):
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

    def run(self, plot=True):
        self.make_grid()
        self.setup_system()
        self.assemble_system()
        self.solve()

        if plot:
            ax = plot_solution(self.points, self.solution, self.triangles)
            ax.set_title("numerical solution")
            plt.show()


if __name__ == '__main__':
    def is_dirichlet(p: np.ndarray):
        return p[1] <= 0


    p = Poisson(2, 1, 200, RightHandSide, NeumannBoundaryValues,
                is_dirichlet, quad_degree=4)
    p.run()
