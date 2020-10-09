import numpy as np

from fem.quadrature import quadrature2D

# This is quadrature points and weights for Gaussian quadrature in 1D,
# on the form (quadrature_point in on [-1, 1], weights)
ONE_DIM_GAUSS = [[[0], [2]],  # 1-point rule
                 [[-np.sqrt(1 / 3), np.sqrt(1 / 3)], [1, 1]],  # 2-point
                 [[-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)],  # 3-point
                  [5 / 9, 8 / 9, 5 / 9]],
                 [[-np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7),  # 4-point
                   -np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
                   np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
                   np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7)],
                  [(18 - np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36,
                   (18 + np.sqrt(30)) / 36, (18 - np.sqrt(30)) / 36]]]

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

    This class integrates can integrate over a
     - cell
         - triangle in 2D (then the points in the list domain_corners are
           in R^2, and are the corners of the triangle). Then dim=2.
         - line segment on real axis in 1D (then the points in domain_corners
           are in R^1, and are the endpoints of the line segment). Then dim=1.
     - boundary
         - edge of a triangle in 2D (then the points triangle_corners are in
           R^2, and are the endpoints of the line segment, which now lies in
           the plane). Then dim=1.
         - edge of a line segment in 1D, so just a single point. Then dim=0.

    So the dim variable refers to the dimension of the integral, dim=1 of
    along a line, dim=2 for an area etc. We should therefore have dim + 1
    points in the domain_corners list, while the dimension of the points
    could be anything (this is referred to as the space dimension, space_dim).
    """
    zetas = None
    weights = None

    # A list of points (a list, event when the point is a value in one
    # dimension), describing the domain as the convex hull of these points.
    domain_corners = []
    jacobi_determinant = 0

    def __init__(self, dim, degree):
        self.dim = dim
        self.degree = degree

        if dim == 0:
            pass

        elif dim == 1:
            if self.degree > len(ONE_DIM_GAUSS):
                raise NotImplementedError("Need more quadrature points and "
                                          "weights for 1D gauss of degree " +
                                          str(degree))
            self.points, self.weights = ONE_DIM_GAUSS[self.degree - 1]

        elif dim == 2:
            # The gauss quadrature points are given in barycentric coordinates.
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
        if self.dim != len(points) - 1:
            raise ValueError("The number of points constituting the domain "
                             "is not corresponding to the dimension.")
        self.domain_corners = points
        space_dim = len(self.domain_corners[0])

        if self.dim == 1:
            # The jacobi determinant in 1D. We just do a change of variable
            # from [-1, 1] to the x-variable (first variable) in R^space_dim.
            # NB: this will always calculate the length of the line if we
            # integrate a constant of value 1. The line segment (b, a) for b
            # > a, will also give the value |a - b|.
            jacobi = 0.5 * np.abs(points[1][0] - points[0][0])
            if space_dim == 1:
                # When we just integrate along some interval, we only need
                # the jacobi determinant to transform the integral from [-1,
                # 1] to [a, b].
                ds = 1
            elif space_dim == 2:
                # In two space dimensions, we integrate over the line ds,
                # so we need a new change of variable from ds to dx. This is
                # just multiplied together with the jacobi determinant.
                delta_x, delta_y = points[1] - points[0]
                # TODO this crashes when the points have the same
                # x-coordinate. Must then do the integration over y.
                line_inc = delta_y / delta_x
                ds = np.sqrt(1 + line_inc ** 2)
            else:
                raise NotImplementedError()

            self.jacobi_determinant = jacobi * ds

        if self.dim == 2:
            a, d = points[1] - points[0]
            b, e = points[2] - points[0]
            # Jacobi determinant times 0.5 just because
            magic = 0.5
            self.jacobi_determinant = np.abs(a * e - b * d) * magic

    def quadrature_point(self, q_index: int) -> np.ndarray:
        """
        Return quadrature point with index q_index in the global coordinate
        system.

        :param q_index: index of the quadrature point.
        :return: a vector of dimension dim.
        """
        if len(self.domain_corners) == 0:
            raise ValueError("Need to run reinit before doing calculations "
                             "on a cell.")

        if self.dim == 0:
            # TODO Integrate over just a single point, so no gauss (just for
            # solving 1D PDE problems).
            return self.domain_corners[0]

        elif self.dim == 1:
            # Transform from the quadrature coordinates on the interval [-1, 1]
            # to points in R^dim.
            assert len(self.domain_corners) == 2
            a = self.domain_corners[0]
            b = self.domain_corners[1]
            q = self.points[q_index]
            return 0.5 * ((b - a) * q + a + b)

        if self.dim == 2:
            return self.domain_corners.T @ self.zetas[q_index]
        else:
            raise NotImplementedError()

    def quadrature(self, points: np.ndarray, g: callable):
        if self.dim == 1:
            pass

        elif self.dim == 2:
            return quadrature2D(*points, Nq=self.degree, g=g)
        else:
            raise NotImplementedError()

    def jacobian(self):
        return self.jacobi_determinant
