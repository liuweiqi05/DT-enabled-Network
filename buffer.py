import numpy as np
import tensorflow as tf
import network as nw
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

num_ue, num_bs, num_s = 3, 2, 2
action_dim = num_ue + 2 * num_s * num_ue
state_dim = num_ue + num_ue + num_ue * num_bs + num_ue * num_s + num_bs + num_bs + 6 * num_ue + 2 * num_bs
critic_lr = 0.002
actor_lr = 0.001
gamma = 0
actor_model = nw.get_actor(state_dim, action_dim)
critic_model = nw.get_critic(state_dim, action_dim)
target_actor = nw.get_actor(state_dim, action_dim)
target_critic = nw.get_critic(state_dim, action_dim)
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64, ):

        self.buffer_capacity = buffer_capacity

        self.batch_size = batch_size

        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, state_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim))

    def record(self, obs_tuple):

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)

            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    sampled_actions = sampled_actions.numpy() + noise

    bandwidth_bound = 20 * np.ones(num_ue)
    ratio_bound = np.ones(num_s * num_ue)
    cp_bound = 32 * np.ones(num_s * num_ue)
    upper_bound = np.concatenate((bandwidth_bound, ratio_bound, cp_bound))
    lower_bound = np.zeros(action_dim)
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action)
    # return [np.squeeze(sampled_actions)]
    # return np.squeeze(sampled_actions)
