import numpy as np
from aso_sol import aso
from icecream import ic
import env_1_2 as env
import noise as ns
from buffer import *

save_path = 'E:/Program Files/Project/Multi/saved_model/'

std_dev = 0.2
tau = 0.005

action_dim = num_ue + 2 * num_s * num_ue
state_dim = num_ue+num_ue+num_ue*num_bs+num_ue*num_s+num_bs+num_bs+6*num_ue+2*num_bs

ou_noise = ns.OUActionNoise(mean=np.zeros(action_dim), std_deviation=float(std_dev) * np.ones(action_dim))

env_1 = env.environment_1(num_ue, num_bs, num_s)

buffer = Buffer(1000, 64)
ep_reward_list = []
avg_reward_list = []

for ep in range(10):
    prev_state = env_1.reset()
    episodic_reward = 0
    for _ in range(10):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        ic(action)
        state, reward, done, info = env_1.step(action)
        ic(state)
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

