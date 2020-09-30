import numpy as np


# TODO bør denne klassen fjernes foreløpig? den gjør jo ingenting.

class FE_Q:
    """
    Denne skal representere Lagrange polynomer på en enhets celle
    TODO finn ut hva vi skal gjøre med bycentriske koordinater i 2D
    """

    def __init__(self, dim, degree):
        """

        :param degree: degree of the basis functions
        """
        self.dim = dim
        self.degree = degree

    @property
    def dofs_per_cell(self):
        if self.dim == 1:
            if self.degree == 1:
                return 2
            elif self.degree == 2:
                return 3
        elif self.dim == 2:
            # TODO formel for dette
            if self.degree == 1:
                return 3
            elif self.degree == 2:
                return 6

        raise NotImplementedError()

    def shape_grad(self, x: np.ndarray):
        pass


if __name__ == '__main__':
    fe = FE_Q(2, 1)
    print(fe.dofs_per_cell)
