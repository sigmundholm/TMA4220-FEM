import numpy as np

from fem.fe_q import FE_Q
from fem.quadrature_lib import QGauss
from fem.triangle import Cell, Face


class FEValuesBase:
    cell: Cell = None

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
        boundary x (a vector) and returns True if this point is on the Dirichlet
        part of the boundary.
        :param update_values:
        :param update_gradients:
        :param update_quadrature_points:
        :param update_JxW_values:
        """
        self.fe: FE_Q = fe
        self.quadrature: QGauss = quadrature

        # TODO these arguments are not in original dealii function, this is
        # should be handled in FE_Q
        self.points = points
        self.edges = edges

        self.is_dirichlet = is_dirichlet

        # Flags for what to update when running FEValues.reinit(cell)
        self.update_values = update_values
        self.update_gradients = update_gradients
        self.update_quadrature_points = update_quadrature_points
        self.update_normal_vectors = update_normal_vectors
        self.update_JxW_values = update_JxW_values

        # Create a map to tell if a dof-index points to a vertex on the
        # boundary.
        self.is_boundary = {edge: True for edge in edges.flatten()}
        for i, p in enumerate(points):
            try:
                self.is_boundary[i]
            except KeyError:
                self.is_boundary[i] = False

        # TODO this is a nasty quickfix
        Face.vertex_at_boundary = self.is_boundary
        Cell.points = points
        Face.points = points

    @property
    def n_quadrature_points(self):
        return self.quadrature.degree

    def quadrature_point_indices(self):
        return list(range(self.n_quadrature_points))

    def dof_indices(self):
        return list(range(len(self.cell.corner_indices)))

    def reinit(self, cell: Cell):
        self.cell: Cell = cell
        self.local2global = {loc: glob for loc, glob in
                             zip(range(len(cell.corner_indices)),
                                 cell.corner_indices)}
        self.fe.reinit(cell, update_values=self.update_values,
                       update_gradients=self.update_gradients)

    def quadrature_point(self, q_index):
        """
        :param q_index: local index of the quadrature point on this cell.
        :return: the quadrature point as a vector (one-dim vector for dim=1).
        """
        return self.quadrature.quadrature_point(q_index)

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
        Return the gradient for the i-th shape function (local index on this
        cell), in the q-th quadrature point. The index of the shape functions
        follows the index in the cell-list (the argument of the reinit
        function).

        :param i: local shape function index
        :param q_index: TODO this variable must be used for higher degree
        shape functions
        :return:
        """
        raise NotImplementedError()

    def JxW(self, q_index) -> float:
        """
        Return the weight for this quadrature point.
        :param q_index: index of the quadrature point on this cell.
        :return:
        """
        return self.quadrature.weights[q_index] \
               * self.quadrature.jacobian()


class FEValues(FEValuesBase):

    def reinit(self, cell: Cell):
        """
        Do the necessary preparations for doing the calculations on the provided
        cell, when the next functions are called.
        :param cell: list of indices in the points list, that corresponds to
        the corners of this cell.
        :return:
        """
        super(FEValues, self).reinit(cell)

        # The quadrature should integrate over the whole cell.
        self.quadrature.reinit(cell.corner_points)

    def shape_value(self, i, q_index) -> float:
        x_q = self.quadrature_point(q_index)
        if self.is_boundary[self.local2global[i]] and self.is_dirichlet(x_q):
            return 0

        return self.fe.shape_value(i, x_q)

    def shape_grad(self, i, q_index) -> np.ndarray:
        factor = 1
        global_index = self.local2global[i]
        if self.is_boundary[global_index] and \
                self.is_dirichlet(self.points[global_index]):
            # TODO this test strongly assumes linear shape functions.
            factor = 0

        return self.fe.shape_grad(i) * factor


class FEFaceValues(FEValuesBase):

    def reinit(self, cell: Cell, face: Face = None):
        super(FEFaceValues, self).reinit(cell)

        # The quadrature should integrate over a face of the cell.
        self.quadrature.reinit(self.cell.corner_points)

    def shape_value(self, i, q_index):
        x_q = self.quadrature.quadrature_point(q_index)
        self.fe.shape_value(i, x_q)
