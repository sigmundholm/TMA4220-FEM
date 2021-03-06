import numpy as np

from fem.fe.fe_q import FE_Q
from fem.fe.fe_system import FESystem
from fem.quadrature_lib import QGauss
from fem.triangle import Cell, Face


class FEValuesBase:
    cell: Cell = None

    # Maps from local to global dof indices.
    loc2glob_dofs = {}
    loc2glob_vertices = {}

    def __init__(self, fe: FE_Q, quadrature: QGauss, points: np.ndarray,
                 edges: np.ndarray, is_dirichlet: callable, update_values=True,
                 update_gradients=False, update_quadrature_points=False,
                 update_normal_vectors=False, update_JxW_values=False):
        """
        :param fe: a single FE_Q object for scalar problems, and a FESystem
        object for vector problems.
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

        self.all_fe_values: [FEValues] = []
        if type(fe) is FESystem:
            for fe_q in fe.fe_qs:
                fe_values = self.__class__(fe_q, quadrature, points, edges,
                                           is_dirichlet, update_values,
                                           update_gradients,
                                           update_quadrature_points,
                                           update_normal_vectors,
                                           update_JxW_values)
                self.all_fe_values.append(fe_values)

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
        return self.quadrature.n

    def quadrature_point_indices(self):
        return list(range(self.n_quadrature_points))

    def dof_indices(self):
        return list(range(self.fe.dofs_per_cell))

    def reinit(self, cell: Cell):
        self.cell: Cell = cell

        self.loc2glob_dofs = self.__create_local2global_dof_map(cell)
        self.loc2glob_vertices = self.__create_local2global_vertex_map(cell)

        self.fe.reinit(cell, update_values=self.update_values,
                       update_gradients=self.update_gradients)

    def __create_local2global_dof_map(self, cell):
        if type(self.fe) is FESystem:
            # Vector problem
            loc2glob = {}

            # Local index
            k = 0
            for global_node_index in cell.corner_indices:
                for d, fe_q in enumerate(self.fe.fe_qs):
                    loc2glob[k] = len(self.fe.fe_qs) * global_node_index + d
                    k += 1
            return loc2glob

        return {loc: glob for loc, glob in zip(range(len(cell.corner_indices)),
                                               cell.corner_indices)}

    def __create_local2global_vertex_map(self, cell):
        if type(self.fe) is FESystem:
            # Vector problem
            loc2glob = {}

            # Local index
            k = 0
            for glob in cell.corner_indices:
                for _ in self.fe.fe_qs:
                    loc2glob[k] = glob
                    k += 1
            return loc2glob

        return {loc: glob for loc, glob in zip(range(len(cell.corner_indices)),
                                               cell.corner_indices)}

    def quadrature_point(self, q_index):
        """
        :param q_index: local index of the quadrature point on this cell.
        :return: the quadrature point as a vector (one-dim vector for dim=1).
        """
        return self.quadrature.quadrature_point(q_index)

    def shape_value(self, i, q_index) -> float:
        """
        Return the value of shape function i evaluated in quadrature point q.
        :param i: local shape function index
        :param q_index: local quadrature point index
        :return:
        """
        x_q = self.quadrature_point(q_index)
        global_index = self.loc2glob_vertices[i]
        shape_center = self.points[global_index]
        if self.is_boundary[global_index] and self.is_dirichlet(shape_center):
            return 0

        # TODO er dette riktig?
        if len(self.loc2glob_dofs.keys()) != len(self.cell.corner_indices):
            i = i // 2
        return self.fe.shape_value(i, x_q)

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
        factor = 1
        global_index = self.loc2glob_vertices[i]
        shape_center = self.points[global_index]
        if self.is_boundary[global_index] and self.is_dirichlet(shape_center):
            # TODO this test strongly assumes linear shape functions.
            factor = 0

        # TODO er dette riktig?
        if len(self.loc2glob_dofs.keys()) != len(self.cell.corner_indices):
            i = i // 2
        return self.fe.shape_grad(i) * factor

    def JxW(self, q_index) -> float:
        """
        Return the weight for this quadrature point.
        :param q_index: index of the quadrature point on this cell.
        :return:
        """
        return self.quadrature.weights[q_index] \
               * self.quadrature.jacobian()

    def __getitem__(self, item):
        """
        Used for vector problems, when fe is of type FESystem.
        :param item:
        :return:
        """
        assert type(self.fe) is FESystem
        return self.all_fe_values[item]


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

        if self.all_fe_values:
            assert type(self.fe) is FESystem
            for fe_v in self.all_fe_values:
                fe_v.reinit(cell)
                fe_v.loc2glob_dofs = self.loc2glob_dofs
                fe_v.loc2glob_vertices = self.loc2glob_vertices

        # The quadrature should integrate over the whole cell.
        self.quadrature.reinit(cell.corner_points)


class FEFaceValues(FEValuesBase):

    def reinit(self, cell: Cell, face: Face = None):
        super(FEFaceValues, self).reinit(cell)

        if self.all_fe_values:
            assert type(self.fe) is FESystem
            for fe_v in self.all_fe_values:
                fe_v.reinit(cell, face)
                fe_v.loc2glob_dofs = self.loc2glob_dofs
                fe_v.loc2glob_vertices = self.loc2glob_vertices

        # The quadrature should integrate over a face of the cell.
        self.quadrature.reinit(face.edge_points)
