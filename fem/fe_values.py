import numpy as np

from fem.fe_q import FE_Q
from fem.quadrature_lib import QGauss


class FE_Values:
    cell = None
    triangle_corners = None

    # List of the constants describing the shape functions that are nonzero
    # on this cell. The functions are linear and on the form
    #   phi(x) = ax + by + c
    # Each element in the list is the vector [a, b, c].
    shape_functions = []
    shape_gradients = None

    global2local = {}
    local2global = {}

    def __init__(self, fe: FE_Q, quadrature: QGauss, points,
                 update_values=True, update_gradients=False,
                 update_quadrature_points=False,
                 update_normal_vectors=False, update_JxW_values=False):
        self.fe = fe
        self.quadrature = quadrature
        self.points = points  # TODO not in original dealii function

        # Flags for what to update when running FEValues.reinit(cell)
        self.update_values = update_values
        self.update_gradients = update_gradients
        self.update_quadrature_points = update_quadrature_points
        self.update_normal_vectors = update_normal_vectors
        self.update_JxW_values = update_JxW_values

    def reinit(self, cell):
        """
        Do the necessary preparations for doing the calculations on the provided
        cell, when the next functions are called.
        :param cell: list of indices in the points list, that corresponds to
        the corners of this cell.
        :return:
        """
        self.cell = cell
        self.triangle_corners = self.points[cell]

        self.global2local = {glob: loc for glob, loc in zip(cell, range(len(
            cell)))}
        print(cell)
        print(self.global2local)

        # Create the matrix to find the coefficients for the shape
        # functions on the current triangle (assumes linear shape
        # functions).
        point_matrix = np.ones((3, 3))
        point_matrix[:, :2] = self.triangle_corners

        for index, k in enumerate(cell):
            rhs = np.zeros(3)
            rhs[index] = 1
            plane_consts = np.linalg.solve(point_matrix, rhs)
            self.shape_functions.append(plane_consts)
            # self.grad_phi.append(plane_consts[:2])

        if self.update_gradients:
            if self.shape_gradients is None:
                self.shape_gradients = []
            for i in range(len((self.cell))):
                self.shape_gradients.append(self.shape_functions[i][:2])

    def quadrature_point_inidices(self):
        raise NotImplementedError()

    def dof_indices(self):
        return list(range(len(self.cell)))

    def quadrature_point(self, q_index):
        """
        :param q_index: index of the quadrature point on this cell.
        :return: the quadrature point as a vector (one-dim vector for dim=1).
        """
        raise NotImplementedError()

    def shape_value(self, i, q_index):
        """
        Return the value of shape function i evaluated in quadrature point q.
        :param i: global index
        :param q_index: local quadrature point index
        :return:
        """
        raise NotImplementedError()

    def shape_grad(self, i, q_index):
        """

        :param i: local shape function index
        :param q_index:
        :return:
        """
        # TODO depends on the quadrature point for higher degree shape
        # functions.
        if self.shape_gradients is None:
            raise Exception("Must set update_gradients flag in reinit call.")
        return self.shape_gradients[i]

    def JxW(self, q_index):
        """

        :param q_index:
        :return:
        """
        # TODO denne må nok tilpasses de Barycentriske koordinatene!!!
