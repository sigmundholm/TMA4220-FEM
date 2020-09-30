from fem.cell import Cell
from fem.fe_q import FE_Q
from fem.tria import Triangulation


class DofHandler:
    """
    Keep track on all the degrees of freedom.
    """
    fe = None

    def __init__(self, triangulation: Triangulation):
        self.triangulation = triangulation

    def distribute_dofs(self, fe: FE_Q):
        self.fe = fe
        # TODO fe vet polynomgraden til basis funksjonene, og kan derfor
        # nummerere dofs. feks for lineære elementer er det èn dof i hvert
        # hjørne av en trekant  (som tilsvarer på endene av elementet i 1D.

    def n_dofs(self):
        """
        Return the number of degrees of freedom. In
        :return:
        """
        if self.fe.degree == 1:
            return len(self.triangulation.points)

        raise NotImplementedError(
            "Mangler å regne dofs for høyere grads basis funksjoner.")

    def active_cell_iterators(self) -> [Cell]:
        """
        Active cells are all the cells the grid is currently split into. If
        before this function is called, we refine the mesh by splitting one of
        the triangles into two, this function will return those two new
        triangles in the returned list instead of the previous one.
        :return: a list of all the triangles the mesh is currently split into.
        """
        return []
