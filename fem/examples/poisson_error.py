from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from fem.examples.poisson import Poisson
from fem.fe_values import FEValues
from fem.function import Function
from fem.plotting import plot_solution
from fem.quadrature_lib import QGauss
from fem.triangle import Cell


class AnalyticalSolution(Function):
    def value(self, p):
        r_sq = (p ** 2).sum()
        return np.sin(2 * np.pi * r_sq)

    def gradient(self, p, value=None):
        r_sq = (p ** 2).sum()
        if value is None:
            value = np.zeros(2)
        kernel = np.sin(2 * np.pi * r_sq)
        value[0] = 4 * np.pi * p[0] * kernel
        value[1] = 4 * np.pi * p[1] * kernel
        return value


class RightHandSide(Function):
    def value(self, p):
        r_sq = (p ** 2).sum()
        return - 8 * np.pi * np.cos(2 * np.pi * r_sq) + 16 * np.pi ** 2 * r_sq \
               * np.sin(2 * np.pi * r_sq)


class NeumannBoundaryValues(Function):
    def value(self, p):
        r_sq = (p ** 2).sum()
        return 4 * np.pi * np.sqrt(r_sq) * np.cos(2 * np.pi * r_sq)


class PoissonError(Poisson):
    def __init__(self, dim, degree, num_triangles, RHS, NeumannBD, is_dirichlet,
                 AnalyticalSoln):
        super().__init__(dim, degree, num_triangles, RHS, NeumannBD,
                         is_dirichlet)
        self.AnalyticalSoln = AnalyticalSoln

    def compute_error(self):
        print("Compute error")
        analytical_soln: Function = self.AnalyticalSoln()

        guass = QGauss(dim=self.dim, degree=self.degree)
        fe_values = FEValues(self.fe, guass, self.points, self.edges,
                             self.is_dirichlet, update_gradients=True)

        l2_diff_integral = 0
        h1_diff_integral = 0
        for triangle in self.triangles:
            cell = Cell(self.dim, triangle)
            fe_values.reinit(cell, )

            for q_index in fe_values.quadrature_point_indices():

                numerical_sol = 0
                numerical_grad = 0
                # Interpolate the solution value and gradient in the quadrature
                # point.
                for i in fe_values.dof_indices():
                    global_index = fe_values.local2global[q_index]
                    numerical_sol += self.solution[global_index] \
                                     * fe_values.shape_value(i, q_index)

                    numerical_grad += self.solution[global_index] \
                                      * fe_values.shape_grad(i, q_index)

                x_q = fe_values.quadrature_point(q_index)
                exact_solution = analytical_soln.value(x_q)
                exact_grad = analytical_soln.gradient(x_q)

                # Integrate the square difference of the two
                l2_diff_integral += (numerical_sol - exact_solution) ** 2 * \
                                    fe_values.JxW(q_index)

                # TODO calculate error in H1-norm
                # Integrate the square difference of the gradients.
                gradient_diff = numerical_grad - exact_grad
                h1_diff_integral += gradient_diff @ gradient_diff * \
                                    fe_values.JxW(q_index)

        l2_error = np.sqrt(l2_diff_integral)
        h1_error = np.sqrt(l2_diff_integral + h1_diff_integral)
        h1_error_semi_norm = np.sqrt(h1_diff_integral)

        Error = namedtuple("Error", ["L2_error", "H1_error", "H1_semi_error"])

        print("L2", l2_error)
        print("H1", h1_error)
        print("H1-semi", h1_error_semi_norm)
        return Error(L2_error=l2_error, H1_error=h1_error,
                     H1_semi_error=h1_error_semi_norm)

    def plot_analytical(self):
        analytical = self.AnalyticalSoln()
        values = []
        for p in self.points:
            values.append(analytical.value(p))
        values = np.array(values)

        ax = plot_solution(self.points, values, self.triangles)
        ax.set_title("analytical")

        ax = plot_solution(self.points, self.solution
                           - values.reshape((-1, 1)),
                           self.triangles)
        ax.set_title("diff = numerical soln - exact soln")

    def run(self):
        super().run()
        self.plot_analytical()
        plt.show()
        return self.compute_error()


if __name__ == '__main__':
    def is_dirichlet(p: np.ndarray):
        return True


    def half_dirichlet(p: np.ndarray):
        return p[1] <= 0


    p = PoissonError(2, 1, 400, RightHandSide, NeumannBoundaryValues,
                     is_dirichlet, AnalyticalSolution)
    p.run()

    p = PoissonError(2, 1, 400, RightHandSide, NeumannBoundaryValues,
                     half_dirichlet, AnalyticalSolution)
    p.run()
