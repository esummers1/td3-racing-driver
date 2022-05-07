import numpy as np
import tensorflow as tf
import keras.backend as k

from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise, BatchNormalization


class Actor:
    EXPLORE_POLICY_FILE = "_actor_working.h5"
    TARGET_POLICY_FILE = "_actor_target.h5"

    LAYER_1_SIZE = 256
    LAYER_2_SIZE = 256

    def __init__(
        self,
        state_size,
        action_size,
        lr,
        tau,
        sigma,
        noise_clip,
        param_noise=True,
        batch_norm=False,
        action_noise=False,
        target_policy_smoothing=False,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.tau = tau
        self.sigma = sigma
        self.noise_clip = noise_clip
        self.param_noise = param_noise
        self.batch_norm = batch_norm
        self.action_noise = action_noise
        self.target_policy_smoothing = target_policy_smoothing
        self.model = self.create_network()
        self.target_model = self.create_network()
        self.adam_optimizer = self.optimizer()
        self.target_model.set_weights(self.model.get_weights())

    def create_network(self):
        state = Input((self.state_size), name="actor_input")
        x = Dense(Actor.LAYER_1_SIZE, activation='relu', name="actor_hidden_1")(state)
        if self.param_noise:
            x = GaussianNoise(self.sigma, name="actor_noise_1")(x)
        if self.batch_norm:
            x = BatchNormalization(name="actor_batchnorm_1")(x)
        x = Dense(Actor.LAYER_2_SIZE, activation='relu', name="actor_hidden_2")(x)
        if self.param_noise:
            x = GaussianNoise(self.sigma, name="actor_noise_2")(x)
        if self.batch_norm:
            x = BatchNormalization(name="actor_batchnorm_2")(x)
        x = Dense(
            self.action_size,
            activation='tanh',
            kernel_initializer=RandomUniform(),
            name="actor_output")(x)
        return Model(state, x)

    def optimizer(self):
        """Creates the optimizer used for gradient ascent."""
        action_grads = k.placeholder(shape=(None, self.action_size))
        params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_grads)
        grads = zip(params_grad, self.model.trainable_weights)
        return k.function(
            inputs=[self.model.input, action_grads],
            outputs=[k.constant(1)],
            updates=[tf.optimizers.Adam(self.lr).apply_gradients(grads)])

    def act(self, state):
        """Decides what to do in a state, according to the exploration policy."""
        action = self.model.predict(np.expand_dims(state, axis=0))[0]
        if not self.action_noise:
            return action
        else:
            noise = self.clipped_noise(len(action))
            return np.clip(action + noise, -self.action_size, self.action_size)

    def act_target(self, state):
        """Decides what to do in a state, according to the target policy."""
        return self.target_model.predict(np.expand_dims(state, axis=0))[0]

    def intent(self, states):
        """Decides what to do in a number of states, according to the exploration policy."""
        return self.model.predict(states)

    def target_intent(self, states):
        """Decides what to do in a number of states, according to the target policy."""
        actions = self.target_model.predict(states)
        if not self.target_policy_smoothing:
            return actions
        else:
            noise = self.clipped_noise(actions.shape)
            return np.clip(actions + noise, -self.action_size, self.action_size)

    def clipped_noise(self, shape):
        return np.clip(self.noise(shape), -self.noise_clip, self.noise_clip)

    def noise(self, shape):
        return np.random.standard_normal(shape) * self.sigma

    def train(self, states, gradients):
        """Trains the exploration policy using a gradient."""
        self.adam_optimizer([states, gradients])

    def train_target(self):
        """Moves the target policy towards the working policy by a small amount."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def save(self, path):
        self.model.save_weights(path + Actor.EXPLORE_POLICY_FILE)
        self.target_model.save_weights(path + Actor.TARGET_POLICY_FILE)

    def load(self, path):
        self.model.load_weights(path + Actor.EXPLORE_POLICY_FILE)
        self.target_model.load_weights(path + Actor.TARGET_POLICY_FILE)
