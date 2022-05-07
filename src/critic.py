import keras.backend as k

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, BatchNormalization


class Critic:
    EXPLORE_FUNCTION_FILE = "_critic_working.h5"
    TARGET_FUNCTION_FILE = "_critic_target.h5"

    LAYER_1_SIZE = 256
    LAYER_2_SIZE = 256

    def __init__(self, state_size, action_size, lr, tau, batch_norm=False):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.tau = tau
        self.batch_norm = batch_norm
        self.model = self.create_network()
        self.target_model = self.create_network()
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')
        self.target_model.set_weights(self.model.get_weights())
        self.action_gradients = k.function(
            [self.model.input[0], self.model.input[1]],
            k.gradients(self.model.output, [self.model.input[1]]))

    def create_network(self):
        """Constructs the network used to model a value function."""
        state = Input((self.state_size), name="critic_state_input")
        action = Input((self.action_size,), name="critic_action_input")
        x = concatenate([state, action], name="critic_input_concatenate")
        x = Dense(Critic.LAYER_1_SIZE, activation='relu', name="critic_hidden_1")(x)
        if self.batch_norm:
            x = BatchNormalization(name="critic_batchnorm_1")(x)
        x = Dense(Critic.LAYER_2_SIZE, activation='relu', name="critic_hidden_2")(x)
        if self.batch_norm:
            x = BatchNormalization(name="critic_batchnorm_2")(x)
        x = Dense(
            1, activation='linear', kernel_initializer=RandomUniform(), name="critic_output")(x)
        return Model(inputs=[state, action], outputs=x)

    def gradients(self, states, actions):
        """Given some states and actions taken from those states, determines the policy gradient."""
        return self.action_gradients([states, actions])

    def target_predict(self, states, actions):
        """Evaluates actions taken from some states using the target function."""
        return self.target_model.predict([states, actions])

    def train(self, states, actions, critic_target):
        """Trains the working function."""
        return self.model.train_on_batch([states, actions], critic_target)

    def train_target(self):
        """Moves the target function towards the working function by a small amount."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def save(self, path):
        self.model.save_weights(path + Critic.EXPLORE_FUNCTION_FILE)
        self.target_model.save_weights(path + Critic.TARGET_FUNCTION_FILE)

    def load(self, path):
        self.model.load_weights(path + Critic.EXPLORE_FUNCTION_FILE)
        self.target_model.load_weights(path + Critic.TARGET_FUNCTION_FILE)
