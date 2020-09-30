import numpy as np


class Triangulation:
    points = None
    triangles = None
    edges = None

    def __init__(self, dim):
        self.dim = dim

