import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from matplotlib.tri import Triangulation
import numpy as np

from fem.supplied import getdisc, getplate


def plot_mesh(points, triangles, edges):
    xs = points[:, 0]
    try:
        ys = points[:, 1]
    except IndexError:
        ys = np.zeros(len(xs))

    fig, ax = plt.subplots()

    ax.scatter(xs, ys)

    # Plot the edges of all triangles
    for triangle in triangles:
        for edge in zip(triangle, list(triangle[1:]) + list([triangle[0]])):
            # TODO very slow..
            try:
                ax.plot(points[edge, 0], points[edge, 1], color="black")
            except IndexError:
                ax.plot(points[edge, 0], [0, 0], color="black")

    for edge in edges:
        ax.plot(points[edge, 0], points[edge, 1], color="pink")

    return ax


def plot_mesh2(points, triangles, latex=True):
    if latex:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        # for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)
    plt.triplot(points[:, 0], points[:, 1])


def plot_solution_old(points, solution, triangles, color="blue", latex=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for triangle in triangles:
        corners = list(points[triangle])
        corners.append(corners[0])

        zs = list(solution[triangle].flatten())
        zs.append(zs[0])

        xs, ys = list(zip(*corners))
        plt.plot(xs, ys, zs, color=color, linewidth=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    return ax


def plot_solution(points, solution, triangles, latex=False):
    if latex:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        # for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)

    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triang = Triangulation(points[:, 0], points[:, 1])

    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tcf = ax1.tricontourf(triang, solution.flatten())
    fig1.colorbar(tcf)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    # ax1.tricontour(triang, z, colors='k')
    return ax1


def plot_vector_field(xs, ys, U_1, U_2, latex=False):
    if latex:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        # for Palatino and other serif fonts use:
        # rc('font',**{'family':'serif','serif':['Palatino']})
        rc('text', usetex=True)

    fig, ax = plt.subplots()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    q = ax.quiver(xs, ys, U_1, U_2)
    ax.quiverkey(q, X=0.3, Y=1.1, U=1,
                 label=r'$\textrm{Arrow length = 1}$', labelpos='E')


def plot_deformed_object(points, U_1, U_2, color="gray"):
    n = int(np.sqrt(len(points)))
    xs, ys = np.linspace(-1, 1, n), np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, ys)

    X += U_1
    Y += U_2

    fig, ax = plt.subplots()

    for i in range(n):
        ax.plot(X[i, :], Y[i, :], color=color)
        ax.plot(X[:, i], Y[:, i], color=color)


if __name__ == '__main__':
    points, tri, edge = getdisc.get_disk(800)
    # plot_mesh2(points, tri)
    points, tri, edge = getplate.get_plate(11)
    print(len(points))
    print(tri)

    print(len(points))

    plot_mesh(points, tri, edge)
    # z = np.linspace(-1, 1, len(points))
    # plot_solution(points, z, tri)
    plt.show()
