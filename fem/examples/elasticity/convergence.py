import numpy as np
import matplotlib.pyplot as plt

from fem.examples.elasticity.elasticity import RightHandSide, Elasticity, \
    AnalyticalSolution, Error
from fem.fe.fe_values import FEValues
from fem.function import Function
from fem.quadrature_lib import QGauss
# from fem.triangle import Cell


class HigherRefinedSolution(Function):
    def __init__(self, problem: Elasticity):
        self.points = problem.points
        u1, u2, U1, U2 = problem.get_as_fields()
        self.U1 = U1
        self.U2 = U2

        n = int(np.sqrt(len(self.points)))
        delta = 2 / (n - 1)
        self.delta = delta

    def value(self, p):
        x, y = p
        index_x = int(round((x + 1) / self.delta, 0))
        index_y = int(round((y + 1) / self.delta, 0))
        return np.array(self.U1[index_x, index_y], self.U2[index_x, index_y])

    def gradient(self, p, value=None):
        return np.array([0, 0])



class Cell2:
    # A list of all the points in the mesh. This is set as a static variable
    # in FEValuesBase.__init__.
    points = []

    def __init__(self, dim, corner_indices):
        self.dim = dim
        self.corner_indices = corner_indices
        self.corner_points = self.points[corner_indices]


class InterpolatedSolution(Function):
    def __init__(self, problem: Elasticity):
        self.problem = problem
        self.points = problem.points
        self.triangles = problem.triangles
        Cell2.points = problem.points

        quad_degree = 4
        gauss = QGauss(dim=2, n=quad_degree)
        self.fe_values = FEValues(self.problem.fe, gauss, self.points,
                                  self.problem.edges, lambda p: True,
                                  update_gradients=True)

        self.cached_gradient_point = None
        self.cached_gradient = None

    def value(self, p):
        for triangle in self.triangles:
            corners = self.points[triangle]
            if self.inside_triangle(p, corners):
                return self.interpolate(p, triangle)
        else:
            raise Exception("Point not inside mesh??")

    def gradient(self, p, value=None):
        if np.all(p == self.cached_gradient_point):
            return self.cached_gradient
        else:
            raise Exception("wut")

    def interpolate(self, p, triangle):

        cell = Cell2(2, triangle)
        self.fe_values.reinit(cell)

        numerical_sol = 0
        numerical_grad = 0

        # Interpolate the solution value and gradient in the quadrature
        # point.
        phis = []
        grad_phis = []

        for k in self.fe_values.dof_indices():
            # This is the global point index

            if k % 2 == 0:
                # k is even
                # Then the test function should be phi_k = [v_k, 0]
                value = [self.fe_values[0].fe.shape_value(k // 2, p), 0]
                grad = [self.fe_values[0].fe.shape_grad(k // 2), [0, 0]]
            else:
                # k is odd
                # Then the test function should be phi_k+1 = [0, v_k]
                value = [0, self.fe_values[1].fe.shape_value(k // 2, p)]
                # TODO ser gradientene riktige ut? har de litt små
                # verdier? skal vel være på størrelsen 1/h
                grad = [[0, 0], self.fe_values[1].fe.shape_grad(k // 2)]
            phis.append(np.array(value))
            grad_phis.append(np.array(grad))

        for i in self.fe_values.dof_indices():
            global_index = self.fe_values.loc2glob_dofs[i]
            numerical_sol += self.problem.solution[global_index] * phis[i]
            numerical_grad += self.problem.solution[global_index] * grad_phis[i]

        self.cached_gradient_point = p
        self.cached_gradient = numerical_grad
        return numerical_sol

    @staticmethod
    def inside_triangle(p, corners):
        indices = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        for i, corner in enumerate(corners):
            a = corners[indices[i][0]]
            b = corners[indices[i][1]]
            res = InterpolatedSolution.half_plane(p, [a, b], corners[i])
            if res < 0:
                return False
        return True

    @staticmethod
    def half_plane(p: np.ndarray, plane, third):
        """
        Returns a positive number if the point p is on the correct side of
        the plane.

        :param p: a point
        :param plane: as list of two numpy points
        :param third: the third point of the triangle
        :return:
        """

        normal = np.array([-(plane[1][1] - plane[0][1]),
                           plane[1][0] - plane[0][0]])
        ac = third - plane[0]
        if normal @ ac < 0:
            # Make sure the normal vector points into the triangle
            normal *= -1

        return normal @ (p - plane[0])


def convergence_against_refined_solution():
    nu = 0.5
    E = 1

    high_n = 128
    rhs = RightHandSide(nu, E)
    analytical = AnalyticalSolution()

    p1 = Elasticity(2, 1, 4, high_n, rhs, analytical, lambda p: True, nu, E)
    p1.run(plot=False)

    # higher_refined = HigherRefinedSolution(p1)
    higher_refined = InterpolatedSolution(p1)

    ns = [32]
    errors = []

    print("\nConvergence loop")
    for n in ns:
        print("n =", n)
        p = Elasticity(2, 1, 4, n, rhs, higher_refined, lambda p: True, nu, E)
        error: Error = p.run(plot=False)

        errors.append((n, error))
        print()
    print()

    for n, e in errors:
        print(n, e)

    print(errors)


def convergence():
    # n = 46 for ca h=0.0625
    # n = 90 for ca h=0.0312
    # ns = [4, 7, 12, 24, 90] for 1/2^i
    ns = [4, 8, 16, 32, 64]
    errors = []

    nu = 1 / 2
    E = 1

    rhs = RightHandSide(nu, E)
    analytical = AnalyticalSolution()

    for n in ns:
        print("n =", n)
        p = Elasticity(2, 1, 4, n, rhs, analytical, lambda p: True, nu, E)
        error: Error = p.run(plot=False)

        errors.append((n, error))
        print()
    print()

    for n, e in errors:
        print(n, e)

    print(errors)


if __name__ == '__main__':
    # convergence_against_refined_solution()
    convergence()
