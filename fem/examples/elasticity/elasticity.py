import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import scipy.sparse.linalg

from fem.fe.fe_system import FESystem
from fem.fe.fe_values import FEValues
from fem.fe.fe_q import FE_Q
from fem.function import Function
from fem.plotting import plot_mesh, plot_solution, plot_solution_old, \
    plot_vector_field, plot_deformed_object
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
        nu = self.nu
        f_1 = sigma * (-2 * y ** 2 - x ** 2 + nu * x ** 2
                       - 2 * nu * x * y - 2 * x * y + 3 - nu)
        f_2 = sigma * (-2 * x ** 2 - y ** 2 + nu * y ** 2
                       - 2 * nu * x * y - 2 * x * y + 3 - nu)

        return np.array([f_1, f_2])


class BoundaryValues(Function):
    def value(self, p):
        return np.array([0, 0])


class AnalyticalSolution(Function):
    def value(self, p):
        x, y = p
        return np.array(
            [(x ** 2 - 1) * (y ** 2 - 1), (x ** 2 - 1) * (y ** 2 - 1)])

    def gradient(self, p, value=None):
        x, y = p
        grad_u1 = np.array([2 * x * (y ** 2 - 1), 2 * y * (x ** 2 - 1)])
        value = np.zeros((2, 2))
        value[0, :] = grad_u1
        value[1, :] = grad_u1
        return value


Error = namedtuple("Error", ["L2_error", "H1_error", "H1_semi_error", "h"])


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

    def __init__(self, dim, degree, quad_degree, num_triangles, rhs: Function,
                 analytical_solution: Function, is_dirichlet: callable, nu, E):
        self.dim = dim
        self.degree = degree
        self.quad_degree = quad_degree
        self.num_triangles = num_triangles

        self.nu = nu
        self.E = E

        self.rhs = rhs
        self.analytical_solution = analytical_solution
        self.fe = FESystem(FE_Q(dim, degree), FE_Q(dim, degree))

        # A function taking a point (np.ndarray) as argument, and returning
        # True if we should have Dirichlet boundary conditions here,
        # and False else.
        self.is_dirichlet = is_dirichlet

    def make_grid(self, plot=True):
        print("Make grid")
        points, triangles, edges = getplate.get_plate(self.num_triangles)
        self.points = points
        self.triangles = triangles
        self.edges = edges

        if plot:
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
        gauss = QGauss(dim=self.dim, n=self.quad_degree)
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
                        value = [fe_values[0].shape_value(k, q), 0]
                        grad = [fe_values[0].shape_grad(k, q), [0, 0]]
                    else:
                        # k is odd
                        # Then the test function should be phi_k+1 = [0, v_k]
                        value = [0, fe_values[1].shape_value(k, q)]
                        # TODO ser gradientene riktige ut? har de litt små
                        # verdier? skal vel være på størrelsen 1/h
                        grad = [[0, 0], fe_values[1].shape_grad(k, q)]
                    phis.append(np.array(value))
                    grad_phis.append(grad)
                assert len(phis) == 6
                assert len(grad_phis) == 6

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

    def solve(self):
        print("Solve")
        sparse_A = scipy.sparse.csr_matrix(self.system_matrix)
        solution = scipy.sparse.linalg.spsolve(sparse_A, self.system_rhs)
        self.solution = solution.reshape((-1, 1))

    def get_as_fields(self):
        u_length = len(self.solution) // 2
        u_1 = np.zeros(u_length)
        u_2 = np.zeros(u_length)

        for i in range(len(self.solution)):
            if i % 2 == 0:
                # i is even, so must be for u_1
                u_1[i // 2] = self.solution[i]
            else:
                u_2[i // 2] = self.solution[i]

        min_x, max_x = -1, 1
        min_y, max_y = -1, 1

        n = int(np.sqrt(len(self.points)))
        U_1 = np.zeros((n, n))
        U_2 = np.zeros((n, n))

        delta_x = ((max_x - min_x) / (n - 1))
        delta_y = ((max_y - min_y) / (n - 1))

        for point, u_1_val, u_2_val in zip(self.points, u_1, u_2):
            x, y = point
            x_index = int(round((x - min_x) / delta_x, 0))
            y_index = int(round((y - min_y) / delta_y, 0))

            U_1[y_index, x_index] = u_1_val
            U_2[y_index, x_index] = u_2_val

        return u_1, u_2, U_1, U_2

    def output_results(self, gradients, plot=True, plot_func=plot_solution,
                       latex=True):
        print("Output results")
        u_1, u_2, U_1, U_2 = self.get_as_fields()

        if plot:
            ax = plot_func(self.points, u_1, self.triangles, latex=latex)
            ax.set_title("numerical u_1")

            ax2 = plot_func(self.points, u_2, self.triangles, latex=latex)
            ax2.set_title("numerical u_2")

            # Plot the displacements as a vector field
            n = int(np.sqrt(len(self.points)))
            xs = np.linspace(-1, 1, n)
            plot_vector_field(xs, xs, U_1, U_2, latex=latex)

            # Visualise the displacements in a deformed object
            plot_deformed_object(self.points, U_1, U_2)

        self.stress_recovery(gradients, plot_func=plot_solution_old, plot=plot)

    def compute_error(self):
        print("Compute error")

        gauss = QGauss(dim=self.dim, n=self.quad_degree)
        fe_values = FEValues(self.fe, gauss, self.points, self.edges,
                             self.is_dirichlet, update_gradients=True)

        # The gradient in each cell
        gradients = []

        l2_diff_integral = 0
        h1_diff_integral = 0
        for triangle in self.triangles:
            cell = Cell(self.dim, triangle)
            fe_values.reinit(cell)

            for q in fe_values.quadrature_point_indices():

                numerical_sol = 0
                numerical_grad = 0

                # Interpolate the solution value and gradient in the quadrature
                # point.
                phis = []
                grad_phis = []

                for k in fe_values.dof_indices():
                    # This is the global point index

                    if k % 2 == 0:
                        # k is even
                        # Then the test function should be phi_k = [v_k, 0]
                        value = [fe_values[0].shape_value(k, q), 0]
                        grad = [fe_values[0].shape_grad(k, q), [0, 0]]
                    else:
                        # k is odd
                        # Then the test function should be phi_k+1 = [0, v_k]
                        value = [0, fe_values[1].shape_value(k, q)]
                        # TODO ser gradientene riktige ut? har de litt små
                        # verdier? skal vel være på størrelsen 1/h
                        grad = [[0, 0], fe_values[1].shape_grad(k, q)]
                    phis.append(np.array(value))
                    grad_phis.append(np.array(grad))

                for i in fe_values.dof_indices():
                    global_index = fe_values.loc2glob_dofs[i]
                    numerical_sol += self.solution[global_index] * phis[i]
                    numerical_grad += self.solution[global_index] * grad_phis[i]

                if q == 0:
                    # Only add the gradient from one quadrature point,
                    # since it should be constant on each cell.
                    gradients.append(numerical_grad)
                else:
                    # Check that the gradient is constant on each cell.
                    assert np.all(gradients[-1] == numerical_grad)

                x_q = fe_values.quadrature_point(q)
                exact_solution = self.analytical_solution.value(x_q)
                exact_grad = self.analytical_solution.gradient(x_q)

                diff = numerical_sol - exact_solution
                # Integrate the square difference of the two
                l2_diff_integral += diff @ diff * fe_values.JxW(q)

                # Integrate rank two tensor product of the gradient difference.
                gradient_diff = numerical_grad - exact_grad
                gradient_diff_ip = np.trace(gradient_diff @ gradient_diff.T)

                h1_diff_integral += gradient_diff_ip * fe_values.JxW(q)

        l2_error = np.sqrt(l2_diff_integral)
        h1_error = np.sqrt(l2_diff_integral + h1_diff_integral)
        h1_error_semi_norm = np.sqrt(h1_diff_integral)

        print("L2", l2_error)
        print("H1", h1_error)
        print("H1-semi", h1_error_semi_norm)
        print("h", self.h)
        return Error(L2_error=l2_error, H1_error=h1_error,
                     H1_semi_error=h1_error_semi_norm, h=self.h), gradients

    def stress_recovery(self, gradients, plot_func=plot_solution, plot=True):
        # Each element: (gradient sum, #elements in the sum)
        nodes = [[0, 0] for i in range(len(self.points))]
        for triangle, gradient in zip(self.triangles, gradients):
            for index in triangle:
                nodes[index][0] += gradient
                nodes[index][1] += 1

        # Divide the gradient sums on the number of elements in each sum to
        # get the average at each node poins.
        avg_gradients = []
        for grad, n in nodes:
            avg_gradients.append(grad / n)

        eps_xx = []
        eps_yy = []
        eps_xy = []
        epsilon_bars = []
        for grad in avg_gradients:
            e_xx = grad[0, 0]
            e_yy = grad[1, 1]
            e_xy = grad[0, 1] + grad[1, 0]

            eps_xx.append(e_xx)
            eps_yy.append(e_yy)
            eps_xy.append(e_xy)

            eps_bar = np.array([e_xx, e_yy, e_xy]).reshape((-1, 1))
            epsilon_bars.append(eps_bar)

        C = self.E / (1 - self.nu ** 2) * np.array([[1, self.nu, 0],
                                                    [self.nu, 1, 0],
                                                    [0, 0, (1 - self.nu) / 2]])
        sigma_bars = []
        sigma_xx = []
        sigma_yy = []
        sigma_xy = []
        for epsilon_bar in epsilon_bars:
            sigma_bar = C @ epsilon_bar
            sigma_bars.append(sigma_bar)

            sigma_xx.append(sigma_bar[0])
            sigma_yy.append(sigma_bar[1])
            sigma_xy.append(sigma_bar[2])

        if plot:
            ax = plot_func(self.points, np.array(sigma_xx), self.triangles,
                           latex=True)
            ax.set_title("$\sigma_{xx}")

            ax = plot_func(self.points, np.array(sigma_yy), self.triangles,
                           latex=True)
            ax.set_title("$\sigma_{yy}")

            ax = plot_func(self.points, np.array(sigma_xy), self.triangles,
                           latex=True)
            ax.set_title("$\sigma_{xy}")

    def run(self, plot=True) -> Error:
        self.make_grid(plot=plot)
        self.setup_system()
        self.assemble_system()
        self.solve()
        error, gradients = self.compute_error()
        self.output_results(gradients, plot=plot, plot_func=plot_solution)

        plt.show()
        return error


if __name__ == '__main__':
    def is_dirichlet(p: np.ndarray):
        return True


    nu = 1 / 2
    E = 1

    rhs = RightHandSide(nu, E)
    analytical = AnalyticalSolution()

    p = Elasticity(2, 1, 3, 10, rhs, analytical, is_dirichlet, nu, E)
    p.run()
