import unittest

import numpy as np
from scipy.integrate import dblquad

from fem.fe_values import FE_Values
from fem.fe_q import FE_Q
from fem.quadrature_lib import QGauss


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
        self.fe_values = FE_Values(fe_q, quadrature,
                                   points=self.triangle_corners,
                                   edges=self.edges,
                                   is_dirichlet=lambda p: False,
                                   update_values=True,
                                   update_gradients=True)
        self.fe_values.reinit(self.triangles[0])

    def test_quadrature_func_on_triangle(self):
        dim = 2

        lower_x = 3
        lower_y = 0
        upper_x = lower_x  # for easy scipy integration
        upper_y = 3

        triangle_corners = np.array([[0, 0],
                                     [lower_x, lower_y],
                                     [upper_x, upper_y]])
        print("corners", triangle_corners)

        def func(p):
            return p[1] ** 3

        for quad_degree in [1, 3, 4]:
            fe_q = FE_Q(dim, degree=1)
            quadrature = QGauss(dim, quad_degree)
            fe_values = FE_Values(fe_q, quadrature,
                                  points=triangle_corners,
                                  edges=self.edges,
                                  is_dirichlet=lambda p: False,
                                  update_values=True,
                                  update_gradients=True)
            fe_values.reinit(self.triangles[0])

            value = 0
            for q_index in range(quad_degree):
                x_q = fe_values.quadrature_point(q_index)
                value += func(x_q) * fe_values.JxW(q_index)

            print("integral-degre", quad_degree, "val", value)

        res = dblquad(lambda x, y: func([y, x]), 0, lower_x,
                      lambda x: lower_y / lower_x * x,
                      lambda x: upper_y / upper_x * x)
        print("INTEGRATED:", res)

    def test_shape_value(self):
        # TODO
        pass
