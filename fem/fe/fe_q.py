import numpy as np

from fem.triangle import Cell


class FE_Q:
    """
    This class represents Lagrange polynomials as basis functions on a cell.
    """
    cell: Cell = None

    # List of the constants describing the shape functions that are nonzero
    # on this cell. The functions are linear and on the form
    #   phi(x) = ax + by + c
    # Each element in the list is the vector [a, b, c].
    shape_functions = []
    shape_gradients = []

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

    def reinit(self, cell: Cell, update_values=True,
               update_gradients=False):
        self.cell = cell
        if update_values:
            self.find_shape_functions()
        if update_gradients:
            self.find_shape_gradients()

    def find_shape_functions(self):
        """
        Create the matrix to find the coefficients for the shape functions on
        the current triangle (assumes linear shape functions).
        """
        self.shape_functions = []  # Empty the list
        point_matrix = np.ones((3, 3))
        point_matrix[:, :2] = self.cell.corner_points

        for index in range(len(self.cell.corner_indices)):
            rhs = np.zeros(3)
            rhs[index] = 1
            plane_consts = np.linalg.solve(point_matrix, rhs)
            self.shape_functions.append(plane_consts)

    def find_shape_gradients(self):
        """
        Use the coefficients for the shape functions to find the gradients.
        """
        self.shape_gradients = []  # Empty the list
        for i in range(len(self.cell.corner_indices)):
            self.shape_gradients.append(self.shape_functions[i][:2])

    def shape_value(self, i, x: np.ndarray):
        if not self.shape_functions:
            raise Exception("Must set update_values flag in reinit call.")

        constants = self.shape_functions[i]
        return constants[:2] @ x + constants[2]

    def shape_grad(self, i):
        # TODO this assumes linear shape functions.
        if not self.shape_gradients:
            raise Exception("Must set update_gradients flag in reinit call.")
        return self.shape_gradients[i]
