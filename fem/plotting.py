import matplotlib.pyplot as plt

from fem.supplied import getdisc


def plot_mesh(points, triangles, edges):
    xs = points[:, 0]
    ys = points[:, 1]

    fig, ax = plt.subplots()

    ax.scatter(xs, ys)

    # Plot the edges of all triangles
    for triangle in triangles:
        for edge in zip(triangle, list(triangle[1:]) + list([triangle[0]])):
            # TODO very slow..
            ax.plot(points[edge, 0], points[edge, 1],
                    color="black")

    return ax


if __name__ == '__main__':
    points, tri, edge = getdisc.get_disk(20)
    plot_mesh(points, tri, edge)
    plt.show()
