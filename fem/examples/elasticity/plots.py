import matplotlib.pyplot as plt
import numpy as np

from fem.convergence_plot import conv_plots
from fem.examples.elasticity.elasticity import Error

if __name__ == '__main__':
    errors = [(5, Error(L2_error=0.23025943788360104, H1_error=1.338395415622962, H1_semi_error=1.3184395624472476, h=0.7071067811865476)), (10, Error(L2_error=0.04633569824556888, H1_error=0.6003694695373366, H1_semi_error=0.5985787358573963, h=0.3142696805273546)), (15, Error(L2_error=0.0192925008443625, H1_error=0.40258230563730446, H1_semi_error=0.40211977347976596, h=0.20203050891044239)), (20, Error(L2_error=0.010157189240128954, H1_error=0.29401952613750143, H1_semi_error=0.29384402879225074, h=0.14886458551295753))]

    ns = []
    hs = []
    l2 = []
    h1 = []
    data = np.zeros((len(errors), 3))
    for i, (n, error) in enumerate(errors):
        ns.append(n)
        hs.append(error.h)
        l2.append(error.L2_error)
        h1.append(error.H1_error)

        data[i, 0] = error.h
        data[i, 1] = error.L2_error
        data[i, 2] = error.H1_error

    print(data)

    conv_plots(data, ["h", r"\|e\|_{L^2}", r"\|e\|_{H^1}"], title="",
               latex=False, domain_size=2, xlabel="$h$")

    plt.show()
