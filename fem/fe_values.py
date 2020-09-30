from fem.fe_q import FE_Q
from fem.qudrature_lib import QGauss


class FE_Values:
    pass

    def __init__(self, fe: FE_Q, quadrature: QGauss,
                 update_values=True, update_gradients=False,
                 update_quadrature_points=False,
                 update_normal_vectors=False, update_JxW_values=False):
        self.fe = fe
        self.quadrature = quadrature

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
        :param cell:
        :return:
        """
        pass

    def quadrature_point_inidices(self):
        raise NotImplementedError()

    def dof_indices(self):
        raise NotImplementedError()

    def quadrature_point(self, q_index):
        """
        :param q_index: index of the quadrature point on this cell.
        :return: the quadrature point as a vector (one-dim vector for dim=1).
        """
        raise NotImplementedError()

    def shape_value(self, i, q_index):
        """
        Return the value of shape function i evaluated in quadrature point q.
        :param i:
        :param q_index:
        :return:
        """
        raise NotImplementedError()

    def shape_grad(self, i, q_index):
        """
        Returnerer
        :return:
        """

        raise NotImplementedError()

    def JxW(self, q_index):
        """

        :param q_index:
        :return:
        """
        # TODO denne må nok tilpasses de Barycentriske koordinatene!!!
