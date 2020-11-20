import datetime
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


if __name__ == '__main__':

    # make grid, assemble, solve system, compute error
    data = [(4, [datetime.timedelta(0, 0, 7642), datetime.timedelta(0, 0, 26239), datetime.timedelta(0, 0, 616), datetime.timedelta(0, 0, 20111)]), (8, [datetime.timedelta(0, 0, 3888), datetime.timedelta(0, 0, 160693), datetime.timedelta(0, 0, 837), datetime.timedelta(0, 0, 114372)]), (16, [datetime.timedelta(0, 0, 15006), datetime.timedelta(0, 0, 702939), datetime.timedelta(0, 0, 3885), datetime.timedelta(0, 0, 526522)]), (32, [datetime.timedelta(0, 0, 61092), datetime.timedelta(0, 3, 231375), datetime.timedelta(0, 0, 59563), datetime.timedelta(0, 2, 534273)]), (64, [datetime.timedelta(0, 0, 289389), datetime.timedelta(0, 13, 933246), datetime.timedelta(0, 0, 875863), datetime.timedelta(0, 9, 550345)])]

    ns = []
    grid = []
    assemble = []
    solve = []
    error = []

    for n, lst in data:
        ns.append(n)
        grid.append(lst[0].total_seconds())
        assemble.append(lst[1].total_seconds())
        solve.append(lst[2].total_seconds())
        error.append(lst[3].total_seconds())

    dofs = []
    for n in ns:
        dofs.append(n ** 2 * 2)

    print(ns)
    print(dofs)
    print(grid)
    print(assemble)
    # bar_chart2(data)

    plt.plot(ns, grid, label=r"\textrm{Make grid}")
    plt.plot(ns, assemble, label=r"\textrm{Assemble system}")
    plt.plot(ns, solve, label=r"\textrm{Solve system}")
    plt.plot(ns, error, label=r"\textrm{Compute error}")

    ns_names = [f"${n}$" for n in ns]
    plt.xticks(ns, ns_names)

    plt.xlabel("$N$")
    plt.ylabel(r"\textrm{Time }$[s]$")

    plt.legend()

    plt.show()

