import numpy as np


class QGauss:
    """
    Used by FEValues for doing the numerical integration.
    """

    def __init__(self, dim, degree):
        self.dim = dim
        self.degree = degree

    def quadrature_points(self):  # TODO ?
        pass


if __name__ == '__main__':
    # Verify with example from 1a)
    pass