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

    # Dirichlet
    # n = [50, 101, 200, 400, 800]
    # h = [0.4405771989655584, 0.2879805069141839, 0.20570036208155995, 0.14026412700374538, 0.09599350230472765]
    # L2 = [1.14168252, 0.4044935, 0.19385404, 0.08174407, 0.04315963]
    # 50 Error(L2_error=array([1.14168252]), H1_error=array([11.6843004]), H1_semi_error=11.628389259058974)
    # 101 Error(L2_error=array([0.4044935]), H1_error=array([7.20167565]), H1_semi_error=7.190307169966566)
    # 200 Error(L2_error=array([0.19385404]), H1_error=array([4.76787276]), H1_semi_error=4.763930237864873)
    # 400 Error(L2_error=array([0.08174407]), H1_error=array([2.98741303]), H1_semi_error=2.9862944494938417)
    # 800 Error(L2_error=array([0.04315963]), H1_error=array([2.20402368]), H1_semi_error=2.2036010642203596)

    # Neumann
    # l2 = [1.09804361, 0.46405342, 0.20519723, 0.08935962, 0.04450886]
    # 50 Error(L2_error=array([1.09804361]), H1_error=array([11.59805947]), H1_semi_error=11.545963957032823)
    # 101 Error(L2_error=array([0.46405342]), H1_error=array([7.14875203]), H1_semi_error=7.133674371488227)
    # 200 Error(L2_error=array([0.20519723]), H1_error=array([4.73239699]), H1_semi_error=4.727946206411688)
    # 400 Error(L2_error=array([0.08935962]), H1_error=array([2.98415491]), H1_semi_error=2.9828166888179615)
    # 800 Error(L2_error=array([0.04450886]), H1_error=array([2.19951207]), H1_semi_error=2.199061689808467)
