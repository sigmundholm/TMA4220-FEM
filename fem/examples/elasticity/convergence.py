from fem.examples.elasticity.elasticity import RightHandSide, Elasticity, \
    AnalyticalSolution, Error


def convergence():
    ns = [5, 10, 15, 20]
    errors = []

    nu = 1 / 2
    E = 1

    rhs = RightHandSide(nu, E)
    analytical = AnalyticalSolution()

    for n in ns:
        p = Elasticity(2, 1, 4, n, rhs, analytical, lambda p: True, nu, E)
        error: Error = p.run(plot=False)

        errors.append((n, error))
        print()
    print()

    for n, e in errors:
        print(n, e)

    print(errors)


if __name__ == '__main__':
    convergence()
