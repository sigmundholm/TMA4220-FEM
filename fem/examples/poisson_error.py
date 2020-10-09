import numpy as np
import matplotlib.pyplot as plt

from fem.examples.poisson import Poisson
from fem.fe_values import FEValues
from fem.function import Function
from fem.quadrature_lib import QGauss
from fem.triangle import Cell

from fem.plotting import plot_solution


class AnalyticalSolution(Function):
    def value(self, p):
        r_sq = (p ** 2).sum()
        return np.sin(2 * np.pi * r_sq)


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

        l2_error = 0
        for triangle in self.triangles:
            cell = Cell(self.dim, triangle)
            fe_values.reinit(cell)

            for q_index in fe_values.quadrature_point_indices():

                numerical_solution = 0
                # Interpolate the solution value in the quadrature point.
                for i in fe_values.dof_indices():
                    global_index = fe_values.local2global[q_index]
                    numerical_solution += self.solution[global_index] \
                                          * fe_values.shape_value(i, q_index)

                x_q = fe_values.quadrature_point(q_index)
                exact_solution = analytical_soln.value(x_q)

                # Integrate the square difference of the two
                l2_error += (numerical_solution - exact_solution) ** 2 * \
                            fe_values.JxW(q_index)

                # TODO calculate error in H1-norm

        print("L2", l2_error)
        return l2_error

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
        plt.show()

    def run(self):
        super().run()
        self.plot_analytical()
        return self.compute_error()


if __name__ == '__main__':
    def is_dirichlet(p: np.ndarray):
        return True


    def half_dirichlet(p: np.ndarray):
        return p[1] <= 0


    # p = PoissonError(2, 1, 400, RightHandSide, is_dirichlet,
    # AnalyticalSolution)
    # p.run()

    p = PoissonError(2, 1, 400, RightHandSide, NeumannBoundaryValues,
                     half_dirichlet, AnalyticalSolution)
    p.run()
