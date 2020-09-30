import numpy as np

from fem.cell import Cell
from fem.dof_handler import DofHandler
from fem.fe_values import FE_Values
from fem.fe_q import FE_Q
from fem.function import Function
from fem.grid_generator import GridGenerator
from fem.grid_out import GridOut
from fem.tria import Triangulation


class RightHandSide(Function):
    def value(self, p: np.ndarray):
        return 1


class BoundaryValues(Function):
    def value(self, p: np.ndarray):
        return 0


class Poisson:
    triangulation: Triangulation = None
    fe: FE_Q = None
    dof_handler: DofHandler = None

    system_matrix = None  # the stiffness matrix
    solution = None  # TODO just a numpy array?
    system_rhs = None  # the right hand side vector

    def __init__(self, dim, degree):
        self.fe = FE_Q(dim, degree)
        self.triangulation = Triangulation(dim)
        self.dof_handler = DofHandler(self.triangulation)

    def make_grid(self):
        pass

    def setup_system(self):
        pass

    def assemble_system(self):
        pass

    def solve(self):
        pass

    def output_results(self):
        pass

    def run(self):
        pass
