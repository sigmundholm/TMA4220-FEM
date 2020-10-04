import numpy as np

from fem.quadrature import quadrature2D

# For Gaussian quadrature in barycentric coordinates. On the form
#   (zeta-coordinates, weights)
ONE_POINT = (np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([1]))
THREE_POINTS = (np.array([[.5, .5, 0], [.5, 0, .5], [0, .5, .5]]),
                np.array([1 / 3, 1 / 3, 1 / 3]))
FOUR_POINTS = (np.array([[1 / 3, 1 / 3, 1 / 3], [3 / 5, 1 / 5, 1 / 5],
                         [1 / 5, 3 / 5, 1 / 5], [1 / 5, 1 / 5, 3 / 5]]),
               np.array([-9 / 16, 25 / 48, 25 / 48, 25 / 48]))


class QGauss:
    """
    Used by FEValues for doing the numerical integration.
    """
    zetas = None
    weights = None

    triangle_corners = None
    jacobi_determinant = 0

    def __init__(self, dim, degree):
        self.dim = dim
        self.degree = degree

        if dim == 2:
            if degree == 1:
                self.zetas, self.weights = ONE_POINT
            elif degree <= 3:
                self.degree = 3
                self.zetas, self.weights = THREE_POINTS
            elif degree == 4:
                self.zetas, self.weights = FOUR_POINTS
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def reinit(self, points):
        self.triangle_corners = points
        a, d = points[1] - points[0]
        b, e = points[2] - points[0]
        # Jacobi determinant times 0.5 just because
        magic = 0.5
        self.jacobi_determinant = np.abs(a * e - b * d) * magic

    def quadrature_point(self, q_index: int):
        """
        Return quadrature point with index q_index in the global coordinate
        system.

        :param points: the vertices of the triangle/cell.
        :param q_index:
        :return:
        """
        if self.dim == 2:
            return self.triangle_corners.T @ self.zetas[q_index]
        else:
            raise NotImplementedError()

    def quadrature(self, points: np.ndarray, g: callable):
        if self.dim == 2:
            return quadrature2D(*points, Nq=self.degree, g=g)
        else:
            raise NotImplementedError()

    def jacobian(self):
        return self.jacobi_determinant


if __name__ == '__main__':
    # Verify with example from 1a)
    pass
    p1 = np.array([1, 0])
    p2 = np.array([3, 1])
    p3 = np.array([3, 2])

    ps = np.array([p1, p2, p3])
    print(ps)

    q_gauss = QGauss(2, 1)
    print()
    print(q_gauss.quadrature_point(ps, 0))
