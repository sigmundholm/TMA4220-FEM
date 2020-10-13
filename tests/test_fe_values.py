import unittest

import numpy as np
from scipy.integrate import dblquad

from fem.fe_values import FEValues
from fem.fe_q import FE_Q
from fem.quadrature_lib import QGauss
from fem.triangle import Cell


class FEValuesTest(unittest.TestCase):
    triangle_corners = np.array([[1, 0], [3, 1], [3, 2]])
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    triangles = np.array([[0, 1, 2]])

    fe_values = None

    def setUp(self):
        dim = 2
        degree = 1
        quad_degree = 1

        fe_q = FE_Q(dim, degree)
        quadrature = QGauss(dim, quad_degree)
        self.fe_values = FEValues(fe_q, quadrature,
                                  points=self.triangle_corners,
                                  edges=self.edges,
                                  is_dirichlet=lambda p: False,
                                  update_values=True,
                                  update_gradients=True)
        cell = Cell(dim, self.triangles[0])
        self.fe_values.reinit(cell)

    def test_gauss_degree_of_exactness_on_triangle(self):
        dim = 2

        lower_x = 3
        lower_y = 0
        upper_x = lower_x  # for easy scipy integration
        upper_y = 3

        triangle_corners = np.array([[0, 0],
                                     [lower_x, lower_y],
                                     [upper_x, upper_y]])

        def first_deg(p):
            return 2 * p[0] + 4 * p[1] - 1

        def second_deg(p):
            return p[0] ** 2 - 2 * p[0] * p[1] - p[1] ** 2 / 2 + 1

        def third_deg(p):
            return 4 * p[1] ** 3 + p[0] ** 2 - 3 * p[1]

        def four_deg(p):
            return p[0] ** 4 - third_deg(p)

        def five_deg(p):
            return p[1] ** 5 - four_deg(p)

        # Number of points to max polynomial degree
        degree_of_exactness = {1: 1, 3: 2, 4: 3, 7: 4}

        message = ""

        for func, degree in [(first_deg, 1), (second_deg, 2), (third_deg, 3),
                             (four_deg, 4), (five_deg, 5)]:
            print("\nFunc", func.__name__)
            integral = dblquad(lambda x, y: func([y, x]), 0, lower_x,
                               lambda x: lower_y / lower_x * x,
                               lambda x: upper_y / upper_x * x)[0]
            print("INTEGRATED:", integral)

            for quad_degree in [1, 3, 4, 7]:
                fe_q = FE_Q(dim, degree=1)
                quadrature = QGauss(dim, quad_degree)
                fe_values = FEValues(fe_q, quadrature,
                                     points=triangle_corners,
                                     edges=self.edges,
                                     is_dirichlet=lambda p: False,
                                     update_values=True,
                                     update_gradients=True)
                cell = Cell(dim, self.triangles[0])
                fe_values.reinit(cell)

                value = 0
                for q_index in range(quad_degree):
                    x_q = fe_values.quadrature_point(q_index)
                    value += func(x_q) * fe_values.JxW(q_index)

                print("gauss points", quad_degree, "polydeg", degree,
                      "value", value)

                try:
                    if degree <= degree_of_exactness[quad_degree]:
                        self.assertAlmostEqual(
                            value, integral, 3,
                            msg=f"A polynomial of degree {degree} should be "
                                f"integrated exactly by a gaussian "
                                f"{quad_degree} point quadrature.")
                    else:
                        self.assertNotEqual(
                            value, integral,
                            msg=f"{quad_degree} point Gauss should not handle "
                                f"polynomial of degree {degree}.")

                except AssertionError as e:
                    message += f"\n{e}"

            if message:
                raise AssertionError(message)

    def test_shape_value(self):
        for i, shape_func_consts in enumerate(
                self.fe_values.fe.shape_functions):
            for j, corner in enumerate(self.triangle_corners):
                value = shape_func_consts[:2] @ corner + shape_func_consts[2]
                if i == j:
                    self.assertAlmostEqual(value, 1, places=10,
                                           msg="Shape function i should be 1 "
                                               "in corner i.")
                else:
                    self.assertAlmostEqual(value, 0, places=10,
                                           msg="Shape function i should be 0 "
                                               "in corner j =! i.")
                value2 = self.fe_values.fe.shape_value(i, corner)
                self.assertAlmostEqual(value, value2,
                                       msg="value and value2 should be equal")
