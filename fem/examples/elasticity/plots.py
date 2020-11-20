import matplotlib.pyplot as plt
import numpy as np

from fem.convergence_plot import conv_plots
from fem.examples.elasticity.elasticity import Error

if __name__ == '__main__':
    # errors = [(4, Error(L2_error=0.3921056437067752,
    # H1_error=1.7410133846373341, H1_semi_error=1.6962844011720561, h=0.9428090415820635)), (7, Error(L2_error=0.10608233708863588, H1_error=0.858193913423862, H1_semi_error=0.8516121950721327, h=0.4714045207910319)), (12, Error(L2_error=0.03310944179211311, H1_error=0.5058415649555192, H1_semi_error=0.5047568263043735, h=0.2571297386132901)), (24, Error(L2_error=0.007070835693765315, H1_error=0.24553648034621528, H1_semi_error=0.24543464804994247, h=0.12297509238026937))]

    errors = [(4, Error(L2_error=0.39266783632094815, H1_error=2.963776486400654, H1_semi_error=2.937649235640707, h=0.9428090415820635)), (8, Error(L2_error=0.10090116224750127, H1_error=3.2927693315350313, H1_semi_error=3.2912229985454893, h=0.40406101782088444)), (16, Error(L2_error=0.022112271996986347, H1_error=3.3579199537990334, H1_semi_error=3.3578471471388975, h=0.18856180831641295)), (32, Error(L2_error=0.01538229212121699, H1_error=3.370635999017502, H1_semi_error=3.3706008993889816, h=0.09123958466923217))]
    # errors = [(4, Error(L2_error=0.39210564370677514,
    # H1_error=1.7410133846373341, H1_semi_error=1.6962844011720561, h=0.9428090415820635)), (8, Error(L2_error=0.07999185071148006, H1_error=0.7731347188634686, H1_semi_error=0.768985433757914, h=0.40406101782088444)), (16, Error(L2_error=0.01736112514054536, H1_error=0.3682381736585333, H1_semi_error=0.3678286882139924, h=0.18856180831641295)), (32, Error(L2_error=0.003732322404437845, H1_error=0.18211869170365516, H1_semi_error=0.1820804427645109, h=0.09123958466923217)), (64, Error(L2_error=0.0009335350283984017, H1_error=0.0895117731217933, H1_semi_error=0.08950690498368354, h=0.04489566864676508))]

    # Error when compared to higher refined INTERPOLATED solution
    errors = [(1, Error(L2_error=0.39199547047971955,
                       H1_error=1.7330727641890646,
                    H1_semi_error=1.6881589845442013, h=0.9428090415820635)),
              (1, Error(L2_error=0.0798538336170642, H1_error=0.779810442657529,
                    H1_semi_error=0.7757110877990526, h=0.40406101782088444)),
              (1, Error(L2_error=0.017199665737560633,
              H1_error=0.3656115047300734,
                    H1_semi_error=0.3652067139162486, h=0.18856180831641295)),
              (1, Error(L2_error=0.0035749351821473025, H1_error=0.193273574829434, H1_semi_error=0.19324050963965156, h=0.09123958466923217))]

    # TODO dette er for å plotte relativ feil når vi bruker eksakt løsning.
    u_analytical_l2 = np.sqrt(512 / 225)
    grad_u_analytical_l2 = np.sqrt(512 / 45)
    u_analytical_h1 = np.sqrt(u_analytical_l2 ** 2 + grad_u_analytical_l2 ** 2)

    print(u_analytical_l2)
    print(u_analytical_h1)
    # errors = errors[1:]

    ns = []
    hs = []
    l2 = []
    h1 = []
    data = np.zeros((len(errors), 3))
    for i, (n, error) in enumerate(errors):
        ns.append(n)
        hs.append(error.h)
        l2.append(error.L2_error / u_analytical_l2)
        h1.append(error.H1_error / u_analytical_h1)

        data[i, 0] = error.h
        data[i, 1] = error.L2_error
        data[i, 2] = error.H1_error

    print(data)
    # data = data[:, :-1]
    print(data)

    ax = conv_plots(data, ["h", r"\|e\|_{L^2}", r"\|e\|_{H^1}"], title="",
                    latex=False, domain_size=2, xlabel="$h$",
                    ylabel=r"$\|\hat{u}-\hat{u}_h\| /\|\hat{u}\| $",
                    desired_order=0,
                    reference_line_offset=.5)

    plt.show()
