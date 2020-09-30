import numpy as np


# Quadrature Nodes and weights hardcoded for Nq = [1,4]
def quadrature1D(Nq, a, b, g):
    if Nq == 1:
        zq = np.array([0.0])
        wq = np.array([2.0])
    elif Nq == 2:
        zq = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        wq = np.array([1.0, 1.0])
    elif Nq == 3:
        zq = np.array([-3 / np.sqrt(5), 0, 3 / np.sqrt(5)])
        wq = [5 / 9, 8 / 9, 5 / 9]
    else:
        zq = np.array([-np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7), -np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7),
                       np.sqrt((3 - 2 * np.sqrt(6 / 5)) / 7), np.sqrt((3 + 2 * np.sqrt(6 / 5)) / 7)])
        wq = np.array(
            [(18 - np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36, (18 - np.sqrt(30)) / 36])

    print("Gauss-Legendre Quadrature Points on [-1,1] ( Nq = ", Nq, "): ", zq)
    print("Corresponding Weights: ", wq)

    # Compute integral of g over [a,b] using affine transformation [-1,1]->[a,b]
    I = ((b - a) / 2) * np.sum(wq * g(((b - a) / 2) * zq + (a + b) / 2))

    return I


# Quadrature Nodes and weights for arbitrary Nq computed by Newton-Raphson iteration
def quadrature1D_anyNq(Nq, a, b, g):
    # Initial guess of roots of n'th order Legendre Polynomial
    z = np.linspace(-0.999, 0.999, Nq)
    zprev = 2 * np.ones(Nq, dtype=float)

    # Newton-Raphson iteration (TOL = 10**(-15))
    while np.amax(np.abs(z - zprev)) > 10 ** (-15):

        # Assign zeroth- and first order Legendre Polynomial
        Pnm1 = np.ones(Nq, dtype=float)
        Pn = z

        # Use Bonnet's Recursive formula to obtain n'th order Legendre Polynomial
        for k in range(1, Nq):
            Pnp1 = ((2 * k + 1) * z * Pn - k * Pnm1) / (k + 1)
            Pnm1 = Pn
            Pn = Pnp1

        # Compute derivative of n'th order polynomial
        Pnd = (Nq / (z ** 2 - 1)) * (z * Pn - Pnm1)

        # Perform Newton-Raphson iterative step
        zprev = z.copy()
        z = zprev - Pn / Pnd

    # Compute weights for Gauss-Legendre quadrature points in [-1,1]
    wq = 2 / ((1 - z ** 2) * (Pnd) ** 2)

    print("Gauss-Legendre Quadrature Points on [-1,1] ( Nq = ", Nq, "): ", z)
    print("Corresponding Weights: ", wq)

    # Compute integral of g over [a,b] using affine transformation [-1,1]->[a,b]
    I = ((b - a) / 2) * np.sum(wq * g(((b - a) / 2) * z + (a + b) / 2))

    return I


# Gauss-Legendre Quadrature points & weights for hardcoded Nq = [1,3,4]
def quadrature2D(p1, p2, p3, Nq, g):
    # Set hardcoded values for quadrature points (in barycentric coordinates) and corresponding weights
    if Nq == 1:
        zeta = np.array([1 / 3, 1 / 3, 1 / 3])
        wq = 1.0
    elif Nq == 3:
        zeta = np.array([[1 / 2, 1 / 2, 0], [1 / 2, 0, 1 / 2], [0, 1 / 2, 1 / 2]])
        wq = np.array([[1 / 3], [1 / 3], [1 / 3]])
    else:
        zeta = np.array([[1 / 3, 1 / 3, 1 / 3], [3 / 5, 1 / 5, 1 / 5], [1 / 5, 3 / 5, 1 / 5], [1 / 5, 1 / 5, 3 / 5]])
        wq = np.array([[-9 / 16], [25 / 48], [25 / 48], [25 / 48]])

    # Collect all x-coordinates of triangle vertices
    px = np.array([p1[0], p2[0], p3[0]])

    # Collect all y-coordinates of triangle vertices
    py = np.array([p1[1], p2[1], p3[1]])

    # Compute x-component of Gauss-Legendre quadrature points
    zqx = zeta.dot(px)
    zqy = zeta.dot(py)

    if Nq == 1:
        I = wq * g(zqx, zqy)
    else:
        I = wq.T @ g(zqx, zqy)

    if type(I) is int:
        return I
    else:
        return I[0]
