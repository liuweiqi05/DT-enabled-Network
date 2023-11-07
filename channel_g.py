import numpy as np


def channel_g(x_ue, y_ue, x_bs, y_bs, num_ue, num_bs):
    H, fc, c = 25, 2e9, 3e8

    d = np.sqrt((x_ue - x_bs) ** 2 + (y_ue - y_bs) ** 2)
    dd = np.sqrt(d ** 2 + H ** 2)
    PLos = np.zeros((num_bs, num_ue))
    PL = np.zeros((num_bs, num_ue))
    for i, n_dd in enumerate(dd):
        for j, n_d in enumerate(n_dd):
            if n_d <= 18:
                PLos[i][j] = 1
            else:
                PLos[i][j] = 18 / n_d + ((n_d - 18) / n_d) * np.exp(1) ** (-n_d / 36)
            PL[i][j] = PLos[i][j] * (28 + 22 * np.log10(fc * n_d / c)) + (1 - PLos[i][j]) * (
                    32.4 + 20 * np.log10(fc * n_d / c))
    G = np.power(10, -PL / 10)
    return G
