from fem.fe.fe_q import FE_Q
from fem.triangle import Cell


class FESystem(FE_Q):
    fe_qs = []

    def __init__(self, *fe_qs: [FE_Q]):
        """
        The list fe_qs can only include FE_Q objects, and not FESystem
        objects. This assumption is for example used in
        FEValuesBase.__create_local2global_dof_map.

        :param fe_qs: a list of FE_Q objects.
        """
        self.fe_qs = fe_qs
        dim = fe_qs[0].dim
        degree = fe_qs[0].degree
        super().__init__(dim, degree)

    @property
    def dofs_per_cell(self):
        total = 0
        for fe_q in self.fe_qs:
            total += fe_q.dofs_per_cell
        return total

    def reinit(self, cell: Cell, update_values=True,
               update_gradients=False):
        for fe_q in self.fe_qs:
            fe_q.reinit(cell, update_values, update_gradients)
