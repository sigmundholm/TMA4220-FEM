from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from fem.supplied import getdisc


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

    return ax


def plot_solution(points, solution, triangles, color="blue"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for triangle in triangles:
        corners = list(points[triangle])
        corners.append(corners[0])

        zs = list(solution[triangle].flatten())
        zs.append(zs[0])

        xs, ys = list(zip(*corners))
        plt.plot(xs, ys, zs, color=color)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    return ax


if __name__ == '__main__':
    points, tri, edge = getdisc.get_disk(20)
    plot_mesh(points, tri, edge)
    plt.show()
