import numpy as np
import matplotlib.pyplot as plt

from fem.examples.elasticity.elasticity import RightHandSide, Elasticity, \
    AnalyticalSolution, Error

from fem.function import Function


class HigherRefinedSolution(Function):
    def __init__(self, problem: Elasticity):
        self.points = problem.points
        u1, u2, U1, U2 = problem.get_as_fields()
        self.U1 = U1
        self.U2 = U2

        n = int(np.sqrt(len(self.points)))
        delta = 2 / (n - 1)
        self.delta = delta

    def value(self, p):
        x, y = p
        index_x = int(round((x + 1) / self.delta, 0))
        index_y = int(round((y + 1) / self.delta, 0))
        return np.array(self.U1[index_x, index_y], self.U2[index_x, index_y])

    def gradient(self, p, value=None):
        return np.array([0, 0])


def convergence_against_refined_solution():
    nu = 0.5
    E = 1

    high_n = 128
    rhs = RightHandSide(nu, E)
    analytical = AnalyticalSolution()

    p1 = Elasticity(2, 1, 4, high_n, rhs, analytical, lambda p: True, nu, E)
    p1.run(plot=False)

    higher_refined = HigherRefinedSolution(p1)

    ns = [4, 8, 16, 32]
    errors = []

    print("\nConvergence loop")
    for n in ns:
        print("n =", n)
        p = Elasticity(2, 1, 4, n, rhs, higher_refined, lambda p: True, nu, E)
        error: Error = p.run(plot=False)

        errors.append((n, error))
        print()
    print()

    for n, e in errors:
        print(n, e)

    print(errors)


def convergence():
    # n = 46 for ca h=0.0625
    # n = 90 for ca h=0.0312
    # ns = [4, 7, 12, 24, 90] for 1/2^i
    ns = [4, 8, 16, 32, 64]
    errors = []

    nu = 1 / 2
    E = 1

    rhs = RightHandSide(nu, E)
    analytical = AnalyticalSolution()

    for n in ns:
        print("n =", n)
        p = Elasticity(2, 1, 4, n, rhs, analytical, lambda p: True, nu, E)
        error: Error = p.run(plot=False)

        errors.append((n, error))
        print()
    print()

    for n, e in errors:
        print(n, e)

    print(errors)


if __name__ == '__main__':
    # convergence_against_refined_solution()
    convergence()
