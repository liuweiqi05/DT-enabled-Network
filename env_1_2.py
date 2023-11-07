import math
import random
import gym
import numpy as np
from aso_sol import aso

num_ue, num_s, num_bs, = 3, 2, 2

rd = np.random.RandomState(345)


class environment_1:
    def __init__(self, num_ue, num_bs, num_s):
        self.t_ue = 9e9 * np.ones(num_ue)
        self.rb_per_ue = np.zeros(num_ue)
        self.ue_aso_in = np.zeros(num_ue * 2)
        self.cp_per_ue = np.zeros(num_s * num_ue)
        self.tsk = rd.uniform(500000, 1000000, (num_s, num_ue))
        self.ue_x = rd.uniform(0, 1000, (1, num_ue))
        self.ue_y = rd.uniform(0, 1000, (1, num_ue))
        self.ue_v_x = rd.uniform(0, 3, (1, num_ue))
        self.ue_v_y = rd.uniform(0, 3, (1, num_ue))
        self.bs_x = np.array([[500], [250]])
        self.bs_y = np.array([[500], [250]])
        self.RB_bs = np.array([100.0, 20.0])
        self.CP_bs = np.array([2 * 2 * 64, 2 * 32.0])
        self.pre_actions = np.zeros(num_ue + 2 * num_s * num_ue)

    def reset(self):
        ue_x, ue_y, ue_v_x, ue_v_y, ue_x, ue_y = self.ue_x.flatten(), self.ue_y.flatten(), self.ue_v_x.flatten(), \
            self.ue_v_y.flatten(), self.ue_x.flatten(), self.ue_y.flatten()
        tsk, bs_x, bs_y = self.tsk.flatten(), self.bs_x.flatten(), self.bs_y.flatten()
        obs = np.concatenate(
            (self.t_ue, self.rb_per_ue, self.ue_aso_in, self.cp_per_ue, self.RB_bs, self.CP_bs, ue_x, ue_y, ue_v_x,
             ue_v_y, ue_x, ue_y, bs_x, bs_y))
        return obs

    def step(self, actions):
        rb_ue = np.array(actions[0:num_ue], float) + np.array(self.pre_actions[0:num_ue], float)

        off_ratio = np.array(actions[num_ue:num_ue + num_s * num_ue], float) + np.array(self.pre_actions[num_ue:num_ue + num_s * num_ue], float)
        ratio = off_ratio.reshape(num_s, num_ue)
        cp_ue_s = np.array(actions[num_ue + num_s * num_ue:], float) + np.array(self.pre_actions[num_ue + num_s * num_ue:], float)
        cp_ue = cp_ue_s.reshape(num_s, num_ue)
        rate, ue_aso, ue_aso_rlt = aso(self.ue_x, self.ue_y, self.bs_x, self.bs_y, num_ue, num_bs, rb_ue, self.RB_bs)
        tsk_ofl = np.sum((ratio * self.tsk), 0)
        t_ofl = tsk_ofl / rate
        t_ofl_cp = np.max(ratio * self.tsk / (cp_ue * 2.3e9), axis=0)
        t_ofl_cp[t_ofl_cp == np.inf] = 9e9
        tsk_loc = self.tsk * (1 - ratio)
        if num_s <= 3:
            t_loc = tsk_loc / 2.02e9
        else:
            t_loc = tsk_loc / (2.02e9 / num_s)
        t_loc_ue = np.max(t_loc, axis=0)

        for i in range(num_ue):
            self.t_ue[i] = max(t_loc_ue[i], t_ofl[i] + t_ofl_cp[i])
        self.t_ue[self.t_ue == np.inf] = 9e9

        rewards = -np.sum(self.t_ue)

        CP_used = np.sum((np.sum(cp_ue, axis=0) * ue_aso_rlt), axis=1)
        RB_used = np.sum(rb_ue*ue_aso_rlt, axis=1)
        if (self.RB_bs - RB_used <= 0).any() or (self.CP_bs-CP_used <= 0).any():
            done = True
        else:
            done = False

        ue_x_new, ue_y_new = self.t_ue * self.ue_v_x + self.ue_x, self.t_ue * self.ue_v_y + self.ue_y
        ue_x_t, ue_y_t = ue_x_new.flatten(), ue_y_new.flatten()

        ue_x, ue_y, ue_v_x, ue_v_y, ue_x, ue_y = self.ue_x.flatten(), self.ue_y.flatten(), self.ue_v_x.flatten(), \
            self.ue_v_y.flatten(), self.ue_x.flatten(), self.ue_y.flatten()
        tsk, bs_x, bs_y = self.tsk.flatten(), self.bs_x.flatten(), self.bs_y.flatten()

        obs = np.concatenate((self.t_ue, rb_ue, ue_aso, cp_ue_s, self.RB_bs, self.CP_bs, ue_x, ue_y, ue_v_x, ue_v_y,
                              ue_x_t, ue_y_t, bs_x, bs_y))

        self.pre_actions = actions

        return obs, rewards, done, {}
