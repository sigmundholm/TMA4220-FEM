# Description:
#   Generate a mesh triangulation of the unit disc.
#
# Arguments:
#   N       Number of nodes in the mesh.
#   theta   Angle of the slice.
#
# Returns:
#   p		Nodal points, (x,y)-coordinates for point i given in row i.
#   tri   	Elements. Index to the three corners of element i given in row i.
#   edge  	Edge lines. Index list to the two corners of edge line i given in row i.
#
#   Author: Kjetil A. Johannessen, Abdullah Abdulhaque
#   Last edit: 07-10-2019


import numpy as np
import scipy.spatial as spsa


def get_slice(N, theta):
    # Controlling the input.
    if N < 3:
        raise RuntimeError("GetSlice", "Error. N >= 3 reguired.")
    elif (theta <= 0) or (theta >= 2 * np.pi):
        raise RuntimeError("GetSlice", "Error. 0 < theta < pi reguired for input.")

    # Approximating design.
    M = np.int(np.floor(np.sqrt(2 * N / theta)))
    r = np.linspace(0, 1, M + 1)
    al = np.floor((theta * M) * r)

    # Fine-tuning to get the right amount of DOF.
    al[0] = 1
    i = M
    while sum(al) > N:
        if al[i] > 0:
            al[i] -= 1
        i -= 1
        if i < 1:
            i = M
    while sum(al) < N:
        al[-1] += 1
    alpha = np.zeros(len(al), dtype=np.int)
    for i in range(0, len(al)):
        alpha[i] = np.int(al[i])

    # Special case, small example.
    if (M == 1) or ((M == 2) and (alpha[1] < 3)):
        i = np.arange(0, N - 1)
        t = (theta / (N - 2)) * i
        p = np.zeros((len(i) + 1, 2))
        for j in range(1, len(i) + 1):
            p[j, :] = [np.cos(t[j - 1]), np.sin(t[j - 1])]
        E = np.array([np.arange(2, N), np.arange(3, N + 1)])
        tri = np.zeros((N - 2, 3), dtype=np.int)
        edge = np.zeros((N, 2), dtype=np.int)
        for j in range(0, N - 2):
            tri[j, 0] = 1
            tri[j, 1] = E[0, j]
            tri[j, 2] = E[1, j]
            edge[j, 0] = E[0, j]
            edge[j, 1] = E[1, j]
        edge[N - 2, 0] = N
        edge[N - 2, 1] = 1
        edge[N - 1, 0] = 1
        edge[N - 1, 1] = 2

    # General case, large example.
    else:
        p = np.zeros((N, 2))
        edge = []
        k = 1
        for i in range(1, M + 1):
            t = 0
            for j in range(0, alpha[i]):
                p[k, :] = [np.cos(t) * r[i], np.sin(t) * r[i]]
                if alpha[i] - 1 != 0:
                    t += theta / (alpha[i] - 1)
                k += 1
            edge += [[k, k - alpha[i]]]
            edge += [[k - alpha[i] - alpha[i - 1] + 1, k - alpha[i] + 1]]
        k = N - alpha[-1] + 1
        for i in range(k, N):
            edge += [[i, i + 1]]
        edge = np.array(edge)
        mesh = spsa.Delaunay(p)
        tri = mesh.simplices

    return p, tri, edge
