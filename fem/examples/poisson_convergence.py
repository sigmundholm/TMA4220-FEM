from fem.examples.poisson_error import AnalyticalSolution, PoissonError, \
    RightHandSide, NeumannBoundaryValues


def convergence():
    def is_dirichlet(p):
        return p[1] <= 0

    ns = [50, 101, 200, 400, 800]
    errors = []
    for n in ns:
        p = PoissonError(2, 4, n, RightHandSide, NeumannBoundaryValues,
                         is_dirichlet, AnalyticalSolution)
        error = p.run(plot=False)
        errors.append((n, error))
    print()
    print()
    for n, e in errors:
        print(n, e)


if __name__ == '__main__':
    convergence()
