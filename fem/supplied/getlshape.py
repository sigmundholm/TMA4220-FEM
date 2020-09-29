# Description:
#   Generate a mesh triangulation on the L-shape domain.
#
# Arguments:
#   N       Number of nodes on each local patch.
#
# Returns:
#   p		Nodal points, (x,y)-coordinates for point i given in row i.
#   tri   	Elements. Index to the three corners of element i given in row i.
#   edge  	Edge lines. Index list to the two corners of edge line i given in row i.
#
#   Author: Abdullah Abdulhaque
#   Last edit: 28-10-2019


import numpy as np
import scipy.spatial as spsa


def GetLShape(N):
    # Controlling the input.
    if N < 2:
        raise RuntimeError("GetLShape", "Error. N >= 2 reguired.")

    # Generating nodal points.
    M = 2 * N - 1
    c = np.linspace(-1, 1, M)
    p = []
    for i in c:
        for j in c:
            if (i <= 0) or ((i > 0) and (j <= 0)):
                p += [[i, j]]
    p = np.array(p)

    # Generating elements.
    mesh = spsa.Delaunay(p)
    tri_temp = mesh.simplices
    tri = []
    for i in tri_temp:
        pts = p[i, :]
        if not np.all(np.mean(p[i, :], axis=0) > 0):
            tri.append(i)
    tri = np.array(tri)

    # Generating edges.
    edge = []
    for i in range(1, M):
        edge += [[i, i + 1]]
    for i in range(1, N):
        edge += [[i * M, (i + 1) * M]]
    for i in range(1, N):
        edge += [[N * M - i + 1, N * M - i]]
    edge += [[(N - 1) * M + N, N * M + N]]
    for i in range(2, N):
        edge += [[N * M + (i - 1) * N, N * M + i * N]]
    for i in range(1, N):
        edge += [[M * N + N * (N - 1) - i + 1, M * N + N * (N - 1) - i]]
    for i in range(2, N):
        edge += [[M * N + N * (N - i) + 1, M * N + N * (N - i - 1) + 1]]
    for i in range(1, N + 2):
        edge += [[M * (N - i + 1) + 1, M * (N - i) + 1]]
    edge = np.array(edge)
    edge -= 1

    return p, tri, edge
