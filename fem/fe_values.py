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

    # Maps from local to global dof indices.
    local2global = {}

    def __init__(self, fe: FE_Q, quadrature: QGauss, points: np.ndarray,
                 edges: np.ndarray, is_dirichlet: callable, update_values=True,
                 update_gradients=False, update_quadrature_points=False,
                 update_normal_vectors=False, update_JxW_values=False):
        """

        :param fe:
        :param quadrature:
        :param points:
        :param edges:
        :param is_dirichlet: a function that takes a point on the
        boundary (x, y) and returns True if this point is on the Dirichlet
        part of the boundary.
        :param update_values:
        :param update_gradients:
        :param update_quadrature_points:
        :param update_normal_vectors:
        :param update_JxW_values:
        """
        self.fe = fe
        self.quadrature = quadrature

        self.points = points  # TODO not in original dealii function
        self.edges = edges

        # Create a map to tell if a dof-index points to a vertex on the
        # boundary.
        self.is_boundary = {edge: True for edge in edges.flatten()}
        for i, p in enumerate(points):
            try:
                self.is_boundary[i]
            except KeyError:
                self.is_boundary[i] = False

        self.is_dirichlet = is_dirichlet

        # Flags for what to update when running FEValues.reinit(cell)
        self.update_values = update_values
        self.update_gradients = update_gradients
        # self.update_quadrature_points = update_quadrature_points
        # self.update_normal_vectors = update_normal_vectors
        # self.update_JxW_values = update_JxW_values

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

        self.local2global = {loc: glob for loc, glob in zip(range(len(cell)),
                                                            cell)}

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

        # Use the coefficients for the shape functions to find the gradients.
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
        :param q_index: local index of the quadrature point on this cell.
        :return: the quadrature point as a vector (one-dim vector for dim=1).
        """
        raise self.quadrature.quadrature_point(self.triangle_corners, q_index)

    def shape_value(self, i, x, y):
        """
        Return the value of shape function i evaluated in quadrature point q.
        :param i: global index
        :param q_index: local quadrature point index
        :return:
        """
        if self.is_boundary[self.local2global[i]] and self.is_dirichlet(x, y):
            return 0

        constants = self.shape_functions
        return constants[0] * x + constants[1] * y + constants[2]

    def shape_grad(self, i, q_index):
        """
        Return the gradient for the i-th shape function (local index on this
        cell), in the q-th quadrature point. The index of the shape functions
        follows the index in the cell-list (the argument of the reinit
        function.

        :param i: local shape function index
        :param q_index: TODO this variable must be used for higher degree
        shape functions
        :return:
        """
        factor = 1
        global_index = self.local2global[i]
        if self.is_boundary[global_index] and self.is_dirichlet(
                *self.points[global_index]):
            print("bdd and dirichlet", global_index, self.points[global_index])
            # TODO this test strongly assumes linear shape functions.
            factor = 0

        if self.shape_gradients is None:
            raise Exception("Must set update_gradients flag in reinit call.")
        return self.shape_gradients[i] * factor

    def JxW(self, q_index):
        """

        :param q_index:
        :return:
        """
        # TODO denne må nok tilpasses de Barycentriske koordinatene!!!
        raise NotImplementedError()
