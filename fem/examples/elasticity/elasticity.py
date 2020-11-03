import numpy as np
import matplotlib.pyplot as plt

from fem.fe.fe_system import FESystem
from fem.fe.fe_values import FEFaceValues, FEValues
from fem.fe.fe_q import FE_Q
from fem.function import Function
from fem.plotting import plot_mesh, plot_solution
from fem.supplied import getdisc
from fem.triangle import Cell
from fem.quadrature_lib import QGauss


class RightHandSide(Function):
    def value(self, p):
        return np.array([1, 1])


class BoundaryValues(Function):
    def value(self, p):
        return np.array([0, 0])


class NeumannBoundaryValues(Function):
    def value(self, p):
        return np.array([0, 0])


class Elasticity:
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
                 is_dirichlet: callable, nu, E):
        self.dim = dim
        self.degree = degree
        self.num_triangles = num_triangles

        self.nu = nu
        self.E = E

        self.RHS = RHS
        self.NeumannBD = NeumannBD
        self.fe = FESystem(FE_Q(dim, degree), FE_Q(dim, degree))

        # A function taking a point (np.ndarray) as argument, and returning
        # True if we should have Dirichlet boundary conditions here,
        # and False else.
        self.is_dirichlet = is_dirichlet

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
        # Vector problem in 2D, so twice the dofs here.≤
        self.n_dofs = len(self.points) * 2

        self.system_matrix = np.zeros((self.n_dofs, self.n_dofs))
        self.system_rhs = np.zeros((self.n_dofs, 1))
        self.solution = np.zeros((self.n_dofs, 1))

    def assemble_system(self):
        """
        Set up the stiffness matrix.
        :return:
        """
        print("Assemble system")
        gauss = QGauss(dim=self.dim, n=self.degree + 1)
        fe_values = FEValues(self.fe, gauss, self.points, self.edges,
                             self.is_dirichlet, update_gradients=True)

        face_gauss = QGauss(dim=self.dim - 1, n=self.degree + 1)
        fe_face_values = FEFaceValues(self.fe, face_gauss, self.points,
                                      self.edges, self.is_dirichlet)

        # TODO only homogeneous Dirichlet supperted
        rhs = self.RHS()
        boundary_values = BoundaryValues()
        neumann_bdd_values = self.NeumannBD()

        nu = self.nu
        sigma = self.E / (1 - self.nu ** 2)
        sigma_2 = (1 - self.nu) / 2

        phis = []
        grad_phis = []

        for triangle in self.triangles:
            # Nå er vi i en celle
            # lar indeksen i punktlista være den globale nummereringen til
            # shape funksjoner
            # TODO build this initialization into the for-loop.
            cell = Cell(self.dim, triangle)
            fe_values.reinit(cell)

            for q in fe_values.quadrature_point_indices():
                x_q = fe_values.quadrature_point(q)
                dx = fe_values.JxW(q)

                for k in range(2 * len(fe_values.dof_indices())):
                    # This is the global point index
                    i_hat = k // 2

                    if k % 2 == 0:
                        # k is even
                        # Then the test function should be phi_k = [0, v_2]
                        value = [0, fe_values[1].shape_value(i_hat, q)]
                        phis.append(np.array(value))

                        grad = [[0, 0], fe_values[1].shape_grad(i_hat, q)]
                        grad_phis.append(grad)
                    else:
                        # k is odd
                        # Then the test function should be phi_k = [v_1, 0]
                        value = [fe_values[0].shape_value(i_hat, q), 0]
                        phis.append(np.array(value))

                        grad = [fe_values[0].shape_grad(i_hat, q), [0, 0]]
                        grad_phis.append(grad)

                for i in fe_values.dof_indices():
                    phi_i = phis[i]
                    grad_phi_i = grad_phis[i]
                    # v_1, v_2 = phi_i
                    grad_v_1, grad_v_2 = grad_phi_i

                    for j in fe_values.dof_indices():
                        # phi_j = phis[j]
                        # u_1, u_2 = phi_j
                        grad_phi_j = grad_phis[j]
                        grad_u_1, grad_u_2 = grad_phi_j

                        val = sigma * (
                                grad_v_1[0] * (grad_u_1[0] + nu * grad_u_2[1])
                                + grad_v_1[1] * (sigma_2 * (grad_u_1[1] +
                                                            grad_u_2[0]))
                                + grad_v_2[0] * (sigma_2 * (grad_u_1[1] +
                                                            grad_u_2[0]))
                                + grad_v_2[1] * (nu * grad_u_1[0] + grad_u_2[1])
                        ) * dx
                        self.system_matrix[fe_values.loc2glob_dofs[i],
                                           fe_values.loc2glob_dofs[j]] += val

                    val = rhs.value(x_q) @ phi_i * dx  # (v, f)
                    self.system_rhs[fe_values.loc2glob_dofs[i]] += val

            for face in cell.face_iterators():
                if not face.at_boundary():
                    continue

                fe_face_values.reinit(cell, face)

                for q in fe_face_values.quadrature_point_indices():
                    x_q = fe_face_values.quadrature_point(q)
                    h = neumann_bdd_values.value(x_q)
                    ds = fe_face_values.JxW(q)

                    for k in fe_values.dof_indices():
                        # This is the global point index
                        i_hat = k // 2

                        if k % 2 == 0:
                            # k is even
                            # Then the test function should be phi_k = [0, v_2]
                            value = [0, fe_face_values[1].shape_value(i_hat, q)]
                            phis.append(np.array(value))

                            grad = [[0, 0],
                                    fe_face_values[1].shape_grad(i_hat, q)]
                            grad_phis.append(grad)
                        else:
                            # k is odd
                            # Then the test function should be phi_k = [v_1, 0]
                            value = [fe_face_values[0].shape_value(i_hat, q)]
                            phis.append(np.array(value))

                            grad = [fe_face_values[0].shape_grad(i_hat, q),
                                    [0, 0]]
                            grad_phis.append(grad)

                    for i in fe_values.dof_indices():
                        phi_i = phis[i]
                        grad_phi_i = grad_phis[i]
                        v_1, v_2 = phi_i
                        grad_v_1, grad_v_2 = grad_phi_i

                        for j in fe_values.dof_indices():
                            # phi_j = phis[j]
                            # u_1, u_2 = phi_j
                            grad_phi_j = phis[j]
                            grad_u_1, grad_u_2 = grad_phi_j

                            val = 0
                            self.system_matrix[fe_face_values.loc2glob_dofs[i],
                                               fe_face_values.loc2glob_dofs[j]] \
                                += val

                        # The Neumann boundary term, should be zero on the
                        # Dirichlet part of the boundary, because
                        # fe_face_values should set the shape functions on
                        # the dofs along that part of the boundary to the
                        # zero function.
                        val = phi_i @ h * ds  # (v, h)
                        self.system_rhs[fe_face_values.loc2glob_dofs[i]] += val

        # This fixes so the matrix is invertible, but could just have
        # removed those dofs that are not a dof from the matrix, so this
        # wouldn't be needed. Would then have to set boundary values in a
        # different way after the system is solved.
        for i, point in enumerate(self.points):
            if fe_values.is_boundary[i] and self.is_dirichlet(point):
                self.system_matrix[2 * i, 2 * i] = 1
                self.system_matrix[2 * i + 1, 2 * i + 1] = 1
                # TODO this doesn't implement lifting functions, so this
                # only works with homogeneous dirichlet boundary conditions.
                g_1, g_2 = boundary_values.value(point)
                self.system_rhs[2 * i] = g_1
                self.system_rhs[2 * i + 1] = g_2

    def set_boundary_conditions(self):
        pass

    def solve(self):
        print("Solve")
        self.solution = np.linalg.solve(self.system_matrix, self.system_rhs)

    def output_results(self, plot):
        print("Output results")

        u_length = len(self.solution) // 2
        u_1 = np.zeros(u_length)
        u_2 = np.zeros(u_length)

        for i in range(len(self.solution)):
            if i % 2 == 0:
                # i is even, so must be for u_1 (????) TODO check this..
                u_1[i // 2] = self.solution[i]
            else:
                u_2[(i - 1) // 2] = self.solution[i]

        if plot:
            ax = plot_solution(self.points, u_1, self.triangles)
            ax.set_title("numerical u_1")

            ax2 = plot_solution(self.points, u_2, self.triangles)
            ax2.set_title("numerical u_2")
            plt.show()

    def run(self, plot=True):
        self.make_grid()
        self.setup_system()
        self.assemble_system()
        self.solve()
        self.output_results(plot)


if __name__ == '__main__':
    def is_dirichlet(p: np.ndarray):
        return p[1] <= 0


    def is_dirichlet(p: np.ndarray):
        return True


    nu = 2
    E = 1

    p = Elasticity(2, 1, 200, RightHandSide, NeumannBoundaryValues,
                   is_dirichlet, nu, E)
    p.run()
