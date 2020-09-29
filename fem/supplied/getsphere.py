# Description:
#   Generate a mesh triangulation of the unit ball.
#
# Arguments:
#   N       Number of nodes in the mesh.
#
# Returns:
#   p		Nodal points, (x,y,z)-coordinates for point i given in row i.
#   tet   	Elements. Index to the four corners of element i given in row i.
#   edge  	Index list of all triangles on the ball boundary.
#
#   Author: Kjetil A. Johannessen, Abdullah Abdulhaque
#   Last edit: 07-10-2019


import numpy as np
import scipy.spatial as spsa


def GetSphere(N):
    # Controlling the input.
    if N < 13:
        print("Error. N >= 13 reguired for input.")
        return

    # Defining auxiliary data.
    M, r, alpha = SphereData(N)

    # Generating the nodal points.
    p = NodalPoints(M, r, alpha)

    # Generating the elements.
    mesh = spsa.Delaunay(p)
    tet = Elements(p, mesh)

    # Generating the boundary elements.
    edge = FreeBoundary(mesh)

    return p, tet, edge


def SphereData(N):
    # Approximating design on sphere.
    M = np.int(np.floor((np.sqrt(np.pi) * np.sqrt(12 * N + np.pi) - 3 * np.pi) / (4 * np.pi)))
    r = np.linspace(0, 1, M + 1)
    TotArea = sum(4 * np.pi * r ** 2)
    al = np.floor(((4 * np.pi * N) / TotArea) * r ** 2)

    # Fine tuning to get the right amount of DOF.
    al[0] = 1
    i = 1
    while sum(al) > N:
        if al[i] > 0:
            al[i] -= 1
        i += 1;
        if sum(al[1:M]) == 0:
            i = M
        elif i > M:
            i = 1
    while sum(al) < N:
        al[-1] += 1
    alpha = np.zeros(len(al), dtype=np.int)
    for i in range(0, len(al)):
        alpha[i] = np.int(al[i])

    return M, r, alpha


def NodalPoints(M, r, alpha):
    # Auxiliary function for generating nodal points.
    p = []
    p += [[0, 0, 0]]
    k = 1
    for i in range(1, M + 1):
        p += Shell(alpha[i], r[i], (2.0 * i * np.pi) / (M + 1))

    return np.array(p)


def Shell(N, rad, alpha):
    # Auxiliary function generating a point cloud on a sphere shell of radius r.
    inc = np.pi * (3.0 - np.sqrt(5))
    off = 2.0 / N
    pts = []
    for k in range(0, N):
        y = (k + 0.5) * off - 1
        R = np.sqrt(1 - y ** 2)
        phi = k * inc + alpha
        pts += [[rad * R * np.cos(phi), rad * y, rad * R * np.sin(phi)]]

    return pts


def Elements(p, mesh):
    # Auxiliary function for generating elements.
    tet_temp = mesh.simplices
    tet = []
    for t in tet_temp:
        x1 = p[t[0], :]
        x2 = p[t[1], :]
        x3 = p[t[2], :]
        x4 = p[t[3], :]
        v1 = x2 - x1
        v2 = x3 - x1
        v3 = x4 - x1
        V6 = np.abs(np.dot(np.cross(v1, v2), v3))
        if V6 >= 10 ** -13:
            tet += [t]

    return np.array(tet)


def FreeBoundary(mesh):
    # Auxiliary function for generating boundary nodes.
    edge = []
    for ind, neigh in zip(mesh.simplices, mesh.neighbors):
        for j in range(4):
            if neigh[j] == -1:
                edge += [[ind[j - 1], ind[j - 2], ind[j - 3]]]

    return np.array(edge)
