import numpy as np
import matplotlib.pyplot as plt

from fem.fe.fe_system import FESystem
from fem.fe.fe_values import FEValues
from fem.fe.fe_q import FE_Q
from fem.function import Function
from fem.plotting import plot_mesh, plot_solution, plot_solution_old
from fem.supplied import getplate
from fem.triangle import Cell
from fem.quadrature_lib import QGauss


class RightHandSide(Function):
    def __init__(self, nu, E):
        self.nu = nu
        self.E = E

    def value(self, p):
        sigma = self.E / (1 - self.nu ** 2)
        x, y = p
        f_1 = sigma * (-2 * y ** 2 - x ** 2 + nu * x ** 2
                       - 2 * nu * x * y - 2 * x * y + 3 - nu)
        f_2 = sigma * (-2 * x ** 2 - y ** 2 + nu * y ** 2
                       - 2 * nu * x * y - 2 * x * y + 3 - nu)

        return np.array([f_1, f_2])


class BoundaryValues(Function):
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

    def __init__(self, dim, degree, num_triangles, rhs, neumann_bd,
                 is_dirichlet: callable, nu, E):
        self.dim = dim
        self.degree = degree
        self.num_triangles = num_triangles

        self.nu = nu
        self.E = E

        self.rhs = rhs
        self.neumann_bd = neumann_bd
        self.fe = FESystem(FE_Q(dim, degree), FE_Q(dim, degree))

        # A function taking a point (np.ndarray) as argument, and returning
        # True if we should have Dirichlet boundary conditions here,
        # and False else.
        self.is_dirichlet = is_dirichlet

    def make_grid(self):
        print("Make grid")
        points, triangles, edges = getplate.get_plate(self.num_triangles)
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

        boundary_values = BoundaryValues()

        nu = self.nu
        sigma = self.E / (1 - self.nu ** 2)
        sigma_2 = (1 - self.nu) / 2

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

                phis = []
                grad_phis = []

                for k in fe_values.dof_indices():
                    # This is the global point index

                    if k % 2 == 0:
                        # k is even
                        # Then the test function should be phi_k = [v_k, 0]
                        value = [fe_values[1].shape_value(k, q), 0]
                        phis.append(np.array(value))

                        grad = [fe_values[1].shape_grad(k, q), [0, 0]]
                        grad_phis.append(grad)
                    else:
                        # k is odd
                        # Then the test function should be phi_k+1 = [0, v_k]
                        value = [0, fe_values[0].shape_value(k, q)]
                        phis.append(np.array(value))

                        # TODO ser gradientene riktige ut? har de litt små
                        # verdier? skal vel være på størrelsen 1/h
                        grad = [[0, 0], fe_values[0].shape_grad(k, q)]
                        grad_phis.append(grad)

                for i in fe_values.dof_indices():
                    phi_i = phis[i]
                    grad_phi_i = grad_phis[i]
                    grad_v_1, grad_v_2 = grad_phi_i

                    for j in fe_values.dof_indices():
                        grad_phi_j = grad_phis[j]
                        grad_u_1, grad_u_2 = grad_phi_j

                        val = sigma * (
                                grad_v_1[0] * (grad_u_1[0] + nu * grad_u_2[1])
                                +
                                grad_v_2[1] * (nu * grad_u_1[0] + grad_u_2[1])
                                +
                                (grad_v_1[1] + grad_v_2[0]) *
                                (sigma_2 * (grad_u_1[1] + grad_u_2[0]))
                        ) * dx
                        self.system_matrix[fe_values.loc2glob_dofs[i],
                                           fe_values.loc2glob_dofs[j]] += val

                    val = self.rhs.value(x_q) @ phi_i * dx  # (v, f)
                    self.system_rhs[fe_values.loc2glob_dofs[i]] += val

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

    rhs = RightHandSide(nu, E)

    p = Elasticity(2, 1, 20, rhs, None,
                   is_dirichlet, nu, E)
    p.run()
