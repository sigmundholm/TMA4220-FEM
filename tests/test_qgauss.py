import unittest

import numpy as np
import scipy.integrate

from fem.quadrature_lib import QGauss, ONE_DIM_GAUSS


class QGaussTest(unittest.TestCase):

    def test_quadrature_const_func(self):
        """
        Test that the are of a triangle is calculated correctly. Just
        integrates a constant function of value 1.
        """
        dim = 2

        # These two are the same points, but in the reversed order the
        # second time. This checks the sign or the result.
        point_setups = [
            np.array([[1, 0], [3, 1], [3, 2]]),
            np.array([[1, 0], [3, 2], [3, 1]])
        ]

        # Test a few random point setups too.
        for i in range(10):
            point_setups.append(np.random.random(6).reshape((3, 2)))

        def triangle_area(points):
            line1 = points[1] - points[0]
            line2 = points[2] - points[0]
            line1_length = np.sqrt((line1 ** 2).sum())
            line2_length = np.sqrt((line2 ** 2).sum())
            angle = np.arccos(line1 @ line2 / (line1_length * line2_length))
            return 0.5 * line1_length * line2_length * np.sin(angle)

        func = 1  # Constant function

        for points in point_setups:
            exact_area = triangle_area(points)

            for quad_degree in [1, 3, 4]:
                quadrature = QGauss(dim, quad_degree)
                quadrature.reinit(points)

                value = 0
                for q_index in range(quad_degree):
                    value += func * quadrature.weights[q_index] \
                             * quadrature.jacobian()

                if value < 0:
                    print("negative!!!", points)
                self.assertAlmostEqual(value, exact_area,
                                       msg="Area of triangle not calculated "
                                           "correctly. Failed on these "
                                           "points: " + str(points))

    def test_quadrature_const_func_one_dim_one_space_dim(self):
        """ Test calculating the length of a line in R^1. """
        func = 1

        for degree in range(1, len(ONE_DIM_GAUSS) + 1):
            # Check 10 random point setups for each quadrature degree.
            for points in np.random.random(2 * 10).reshape(-1, 2, 1):
                points *= 10

                gauss = QGauss(dim=1, degree=degree)
                gauss.reinit(points)

                value = 0
                for q_index in range(degree):
                    value += func * gauss.weights[q_index] * gauss.jacobian()
                exact_value = np.abs(points[1][0] - points[0][0])

                self.assertAlmostEqual(value, exact_value, places=10,
                                       msg="Not integrating a constant "
                                           "correctly in one space dim. "
                                           "Points = " + str(points))

    def test_quadrature_const_func_one_dim_two_space_dims(self):
        """ Test calculating the length of a line in R^2. """
        func = 1

        for degree in range(1, len(ONE_DIM_GAUSS) + 1):
            # Check 10 random point setups for each quadrature degree.
            for points in np.random.random(4 * 10).reshape(-1, 2, 2):
                points *= 10
                gauss = QGauss(dim=1, degree=degree)
                gauss.reinit(points)

                value = 0
                for q_index in range(degree):
                    value += func * gauss.weights[q_index] * gauss.jacobian()
                delta_x, delta_y = points[1] - points[0]
                exact_value = np.sqrt(delta_x ** 2 + delta_y ** 2)

                self.assertAlmostEqual(value, exact_value, places=10,
                                       msg="Not integrating a constant "
                                           "correctly in one space dim. "
                                           "Points = " + str(points))

    def test_quadrature_polynomial_one_dim_one_space_dim(self):
        """ Check degree of exacness for Gauss quadrature along a line in R^1"""
        poly1 = lambda x: 3 * x + 4
        poly2 = lambda x: x ** 2 + 5 * x - 7
        poly3 = lambda x: 4 * x ** 3 - 8 * x ** 2 - 3 * x + 1
        poly4 = lambda x: x ** 4 - 4 * x ** 3 - 8 * x ** 2 - 3 * x + 1
        poly5 = lambda x: x ** 5 - 4 * x ** 3 - 8 * x ** 2 - 3 * x + 1

        for func, poly_deg in [(poly1, 1), (poly2, 2), (poly3, 3),
                               (poly4, 4), (poly5, 5)]:
            for degree in range(1, len(ONE_DIM_GAUSS) + 1):
                for points in np.random.random(2 * 5).reshape(-1, 2, 1):
                    points *= 10

                    gauss = QGauss(dim=1, degree=degree)
                    gauss.reinit(points)

                    value = 0
                    for q_index in range(degree):
                        x_q = gauss.quadrature_point(q_index)
                        value += func(x_q[0]) * gauss.weights[q_index] \
                                 * gauss.jacobian()
                    exact_value = scipy.integrate.quad(func, a=min(points),
                                                       b=max(points))[0]

                    if poly_deg <= 2 * degree - 1:
                        self.assertAlmostEqual(
                            value, exact_value, places=10,
                            msg=f"Expect {degree} point Gauss to integrate "
                                f"{poly_deg}. degree polynom exactly."
                                "Points = " + str(points))
