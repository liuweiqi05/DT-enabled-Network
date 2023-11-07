from pulp import *
import numpy as np
import channel_g as c_g

Pm = 200


# Noise = 4e-15 * actions[0] * 180e3
# R = 180e3 * actions[0] * np.log2(1 + self.Pm * G / Noise)
def aso(x_ue, y_ue, x_bs, y_bs, num_ue, num_bs, rb_ue, RB_bs):
    G = c_g.channel_g(x_ue, y_ue, x_bs, y_bs, num_ue, num_bs)
    ue_aso = LpProblem(sense=LpMaximize)
    bs_ue = [[LpVariable(name=f"bs_ue_{i}_{j}", cat=LpBinary) for j in range(num_ue)] for i in range(num_bs)]
    Noise = 4e-15 * 180e3

    profit = lpSum(
        180e3 * bs_ue[i][j] * rb_ue[j] * np.log2(1 + Pm * G[i][j] / Noise * rb_ue[j]) for i in range(num_bs) for j
        in range(num_ue))
    ue_aso += profit

    if num_bs > 1:
        for j in range(num_ue):
            aso_cons = [bs_ue[i][j] for i in range(num_bs)]
            ue_aso += lpSum(aso_cons) <= 1

    for i in range(num_bs):
        ue_aso += lpSum(bs_ue[i][j] * rb_ue[j] for j in range(num_ue)) <= RB_bs[i]

    status = ue_aso.solve(PULP_CBC_CMD(msg=False))

    bs_ue_sol = [bs_ue[i][j].value() for i in range(num_bs) for j in range(num_ue)]

    for i in range(num_bs):
        for j in range(num_ue):
            if bs_ue[i][j].varValue == 1:
                print(f"UE {j} is associated with BS {i}")

    bs_ue_rlt = np.array(bs_ue_sol).reshape((num_bs, num_ue))
    ue_rate = [180e3 * bs_ue_rlt[i][j] * rb_ue[j] * np.log2(1 + Pm * G[i][j] / Noise * rb_ue[j]) for i in
               range(num_bs) for j in range(num_ue)]
    ue_rate = np.array(ue_rate).reshape((num_bs, num_ue))
    ue_rate = np.sum(ue_rate, 0)

    return ue_rate, bs_ue_sol, bs_ue_rlt  # , ue_aso.objective.value()
