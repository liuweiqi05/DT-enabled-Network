import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

num_ue, num_bs, num_s = 3, 2, 2


def get_actor(state_dim, action_dim):
    # Initialize weights between -3e-5 and 3-e5
    last_init = tf.random_uniform_initializer(minval=-0.00003, maxval=0.00003)

    inputs = layers.Input(shape=(state_dim,))
    layer_1 = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(inputs)
    out = layers.Dropout(rate=0.5)(layer_1)
    out = layers.BatchNormalization()(out)
    layer_2 = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")(out)
    out = layers.Dropout(rate=0.5)(layer_2)
    out = layers.BatchNormalization()(out)

    outputs = layers.Dense(action_dim, activation="sigmoid", kernel_initializer=last_init)(out)

    bandwidth_bound = 20 * np.ones(num_ue)
    ratio_bound = np.ones(num_s * num_ue)
    cp_bound = 32 * np.ones(num_s * num_ue)
    upper_bound = np.concatenate((bandwidth_bound, ratio_bound, cp_bound))

    outputs = layers.Lambda(lambda x: x * upper_bound)(outputs)
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(state_dim, action_dim):
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    state_input = layers.Input(shape=state_dim)
    state_out = layers.Dense(32, activation="selu", kernel_initializer="lecun_normal")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(64, activation="selu", kernel_initializer="lecun_normal")(state_out)
    state_out = layers.BatchNormalization()(state_out)

    action_input = layers.Input(shape=action_dim)
    action_out = layers.Dense(64, activation="selu", kernel_initializer="lecun_normal")(action_input)
    action_out = layers.BatchNormalization()(action_out)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(512, activation="selu", kernel_initializer="lecun_normal")(concat)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(512, activation="selu", kernel_initializer="lecun_normal")(out)
    out = layers.Dropout(rate=0.5)(out)
    out = layers.BatchNormalization()(out)

    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)

    return model
