# Description:
#   Removes all nodal points not appearing in the element array.
#
# Arguments:
#   p		Nodal points (matrix of size (Npts x 4)).
#	tri		Element array (matrix of size (Nel x 3) or (Nel x 5)).
#	edge	Boundary elements (matrix of size (Nel x 2) or (Nel x 4)).
#
# Returns:
#   p		The stripped down nodal array (smaller size).
#   tri		Updates nodal indexes of the element array (updated values).
#   edge	Updates nodal indexes of the boundary elements (updated values).
#
#   Author: Kjetil A. Johannessen, Abdullah Abdulhaque
#   Last edit: 07-10-2019


import numpy as np


def remove_unused_2d(p, tri, edge):
    tmp = tri[:, 0:3]
    N = np.shape(p)[0]
    used = np.zeros(N)
    used[tmp.T.flatten()] = 1
    uu = []
    k = -1
    for i in range(0, N):
        k += 1
        if used[k] == 1:
            uu += [k]
    uu = np.sort(np.array(uu))

    Offset = np.zeros(N, dtype=np.int)
    TotOffset = 0
    k = 1
    for i in range(1, N + 1):
        if (k <= len(uu)) and (uu[k - 1] != i - 1):
            TotOffset += 1
        else:
            k += 1
        Offset[i - 1] = np.int(TotOffset)

    p = p[uu, :]
    Nel = np.shape(tri)[0]
    for i in range(0, Nel):
        tri[i, 0:3] -= Offset[tri[i, 0:3]]
    Nel = np.shape(edge)[0]
    for i in range(0, Nel):
        edge[i, 0:2] -= Offset[edge[i, 0:2] - 1]

    return p, tri, edge
