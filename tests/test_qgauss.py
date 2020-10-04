import unittest

import numpy as np

from fem.quadrature_lib import QGauss


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
