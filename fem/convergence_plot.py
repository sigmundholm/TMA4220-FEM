import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import numpy as np

import os


def convergence_plot(ns, errors, yscale="log2", desired_order=2,
                     reference_line_offset=0.5,
                     xlabel="$N$", ylabel=r"\textrm{Error}", title="",
                     show_conv_order=True):
    # Remove small tick lines on the axes, that doesnt have any number with them.
    matplotlib.rcParams['xtick.minor.size'] = 0
    matplotlib.rcParams['xtick.minor.width'] = 0
    matplotlib.rcParams['ytick.minor.size'] = 0
    matplotlib.rcParams['ytick.minor.width'] = 0

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

    res = np.polyfit(np.log(ns), np.log(errors), deg=1)
    print("Polyfit:", res)
    print("Order of convergence", -res[0])

    fig, ax = plt.subplots()

    ax.plot(ns, errors, "-o")
    if yscale == "log2":
        ax.set_yscale("log", basey=2)
    else:
        ax.set_yscale("log")

    ax.set_xscale("log")
    if show_conv_order:
        ax.set_title(
            r"\textrm{" + title + r"}\newline \small{\textrm{Convergence order: " + str(
                -round(res[0], 2)) + " (lin.reg.)}}")
    # title(r"""\Huge{Big title !} \newline \tiny{Small subtitle !}""")

    # Remove scientific notation along x-axis
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xticks(ns)
    ns_names = []
    for n in ns:
        ns_names.append(f'${n}$')
    ax.set_xticklabels(ns_names)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Create reference line for desired order of convergence
    if desired_order:
        ns = np.array(ns)
        # line = desired_order * ns + res[1] - 0.5
        line = np.exp(
            desired_order * np.log(ns) + res[1] - reference_line_offset)
        ax.plot(ns, line, "--", color="gray",
                label=r"$\textrm{Convergence order " + str(
                    desired_order) + " reference}$")
        ax.legend()

    plt.show()


def plot3d(field, title="", latex=False, z_label="z", xs=None, ys=None):
    if latex:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        # for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)

    ys = ys if ys is not None else xs
    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, field, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel("$x$" if latex else "x")
    ax.set_ylabel("$y$" if latex else "y")
    ax.set_zlabel(f"${z_label}$" if latex else z_label)
    return ax


if __name__ == '__main__':
    # Example data with 2nd order convergence
    ns = [10, 20, 40, 80]
    errors = np.array([0.0008550438272162397, 0.00021361488748972146,
                       5.1004121790487744e-05,
                       1.0208028014102588e-05]) / 1.0788731827251923

    ns = [50, 100, 200, 400, 800]
    hs = [0.44, 0.29, 0.21, 0.14, 0.10]
    l2_dirichlet = np.array([1.14168252, 0.4044935, 0.19385404, 0.08174407,
                             0.04315963]) / 2.5911  # Dirichlet
    l2_neumann = np.array([1.09804361, 0.46405342, 0.20519723, 0.08935962,
                           0.04450886]) / 2.5911  # Neumann
    # Neumann

    convergence_plot(hs, l2_neumann, yscale="log10", desired_order=2,
                     reference_line_offset=1, xlabel="$h$",
                     ylabel=r"$\|u-u_h\|_{L^2}/\|u\|_{L^2}$",
                     show_conv_order=False)
    plt.show()
