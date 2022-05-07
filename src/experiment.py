import gym
import time
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from pathlib import Path
from actor import Actor
from critic import Critic
from replay_buffer import ReplayBuffer
from utils import render_time, render_int, datetime_now

# noinspection PyUnresolvedReferences
import gym_torcs


class Experiment:
    """
    Allows for a self-contained experiment. Responsible for coordinating the setup and teardown of
    the environment, and the interactions between the parts of the agent as each episode is run.

    Strictly speaking, this class shouldn't be responsible for the configuration, reward function
    **and** training loop, since the last isn't TORCS-specific, but that isn't a priority.
    """

    def __init__(
        self,
        load_from=None,
        save_as=None,
        train=True,
        render=False,
        training_steps=1e6,
        lap_limit=20,
        step_limit=5e3,
        episodes_per_validation=10,
        batch_size=256,
        buffer_size=None,
        gamma=0.99,
        actor_lr=1e-4,
        critic_lr=1e-3,
        delay=2,
        tau=1e-3,
        sigma=0.2,
        noise_clip=0.5,
        param_noise=False,
        batch_norm=False,
        action_noise=True,
        target_policy_smoothing=True,
        track="road/g-track-2",
        replay_buffer=None,
        td3=True,
    ):
        """
        :param load_from: The folder to load weights from
        :param save_as: The folder to save weights and results to
        :param train: Whether to run the model in training mode (false => test only)
        :param render: Whether to render the simulation (8-10x slower than not rendering)
        :param training_steps: The total number of training steps to take
        :param lap_limit: The number of laps to limit any training or validation episode to
        :param step_limit: The number of steps to limit any training or validation episode to
        :param episodes_per_validation: The number of episodes to perform between validations
        :param batch_size: The number of experiences to sample from the buffer at each step
        :param buffer_size: The number of experiences to store in the buffer before evicting old
                            ones. Will default to the total number of training steps.
        :param gamma: The discount factor for the Bellman equation, for reward value estimation
        :param actor_lr: The initial learning rate for the actor
        :param critic_lr: The initial learning rate for the critics
        :param delay: The number of steps to take for each policy/target update (TD3 only)
        :param tau: The soft update rate for the target networks
        :param sigma: The noise scale factor used for exploration
        :param noise_clip: The maximum amount of perturbation noise will introduce
        :param param_noise: Whether to include parameter space noise in the actor during training
        :param batch_norm: Whether to employ batch normalization on the actor and critic
        :param action_noise: Whether to include noise in the actor's decisions when exploring
        :param target_policy_smoothing: Whether to include noise in, i.e. smooth, the target policy
        :param track: Which track to run the experiment on
        :param replay_buffer: Use this to share replay buffers between experiments
        :param td3: Whether to use TD3 enhancements
        """
        self.load_from = load_from
        self.save_as = save_as
        self.port = 0
        self.train = train
        self.render = render
        self.training_steps = int(training_steps)
        self.lap_limit = int(lap_limit)
        self.step_limit = int(step_limit)
        self.episodes_per_validation = episodes_per_validation
        self.batch_size = batch_size
        self.buffer_size = int(buffer_size) if buffer_size is not None else self.training_steps
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_delay = delay
        self.tau = tau
        self.sigma = sigma
        self.noise_clip = noise_clip
        self.param_noise = param_noise
        self.batch_norm = batch_norm
        self.action_noise = action_noise
        self.target_policy_smoothing = target_policy_smoothing if td3 else False
        self.track_name = track
        self.buffer = ReplayBuffer(self.buffer_size) if replay_buffer is None else replay_buffer
        self.td3 = td3

        # Lazy fields that will get populated during `prepare`
        self.environment = None
        self.actor = None
        self.critics = []
        self.act_dim = None
        self.save_path = None
        self.load_path = None

    def prepare(self):
        """
        Performs the more heavyweight setup actions, setting our lazy fields.

        The main reason for this is that we need to create and observe the environment to create our
        actor and critic, and doing this from the constructor does not allow experiments to be
        queued.
        """

        self.environment = gym.make(
            "Torcs-v0",
            vision=False,
            throttle=True,
            gear_change=False,
            rendering=self.render,
            hard_reset_interval=50,
            rank=self.port,
            lap_limiter=self.lap_limit)

        initial_state = get_state_from(self.environment.reset())
        action_space = self.environment.action_space
        self.act_dim = action_space.high.shape[0]

        self.actor = Actor(
            state_size=initial_state.shape,
            action_size=self.act_dim,
            lr=self.actor_lr,
            tau=self.tau,
            sigma=self.sigma,
            noise_clip=self.noise_clip,
            param_noise=self.param_noise,
            batch_norm=self.batch_norm,
            action_noise=self.action_noise,
            target_policy_smoothing=self.target_policy_smoothing)

        # Create critic(s)
        num_critics = 2 if self.td3 else 1
        for i in range(0, num_critics):
            self.critics.append(Critic(
                state_size=initial_state.shape,
                action_size=self.act_dim,
                lr=self.critic_lr,
                tau=self.tau))

        # Set up save directory
        if self.save_as is None:
            self.save_as = "untitled{}".format(self.port)
        save_to_folder = "../agents/{}".format(self.save_as)
        self.save_path = "{}/{}".format(save_to_folder, self.save_as)
        Path(save_to_folder).mkdir(parents=True, exist_ok=True)

        # Load weights from files, if configured to
        if self.load_from is not None:
            read_from_folder = "../agents/{}".format(self.load_from)
            Path(read_from_folder).mkdir(parents=True, exist_ok=True)
            self.load_path = "{}/{}".format(read_from_folder, self.load_from)

            try:
                self.actor.load(self.load_path)
                for i, critic in enumerate(self.critics, start=1):
                    critic.load(self.load_path + str(i))
                print("\nLoaded weight files from \"{}\" successfully.".format(self.load_from))
            except OSError as _:
                print("\nNo weight files for \"{}\" were found.".format(self.load_from))

    def run(self):
        """Runs the experiment according to the config, saving results to files."""

        start_time = time.time()
        finished = False
        training_steps = 0
        validation_steps = 0
        training_episodes = -1
        steps_taken = 0
        performance = []

        self.prepare()

        progress_bar = tqdm(total=self.training_steps, leave=True, unit=" steps")
        progress_bar.set_description(f"{datetime_now()} | Performance: 0 | Progress")
        while not finished:

            if self.train:
                steps_taken = self.run_training_episode(training_steps)
                training_steps += steps_taken
                training_episodes += 1
                if training_steps == self.training_steps:
                    finished = True
            else:
                finished = True

            if self.should_validate(training_episodes):
                new_performance, validation_steps_taken = self.run_validation_episode()
                performance.append(new_performance)
                validation_steps += validation_steps_taken

            progress_bar.set_description(
                f"{datetime_now()} | Performance: {render_int(performance[-1])} | Progress")
            progress_bar.update(steps_taken)
            progress_bar.refresh()

        self.environment.end()
        duration = time.time() - start_time

        self.save_results(
            performance,
            duration,
            training_steps,
            validation_steps,
            training_episodes)

    def should_validate(self, episode):
        return not self.train or episode == 0 or episode % self.episodes_per_validation == 0

    def run_training_episode(self, prior_steps):
        """Main training loop, training actor and critic(s) on sampled experiences."""

        steps, finished = 0, False
        old_state = get_state_from(self.environment.reset())

        while not finished \
                and steps < self.step_limit \
                and steps + prior_steps < self.training_steps:

            # Act against environment and observe result
            action = self.actor.act(old_state)
            observation, _, finished, _ = self.environment.step(action)
            reward = calculate_reward(observation)
            new_state = get_state_from(observation)
            self.buffer.add(old_state, action, reward, finished, new_state)

            # Sample experience buffer
            states, actions, rewards, finisheds, new_states = self.sample_batch()

            # Predict target q-values using target networks
            actor_intent = self.actor.target_intent(new_states)
            q_values = self.q_value(new_states, actor_intent)
            critic_target = self.bellman(rewards, q_values, finisheds)

            # Train both networks on sampled batch, update target networks
            self.train_critic(states, actions, critic_target)
            actions = self.actor.intent(states)
            policy_gradient = self.gradients(states, actions)

            # Update actor and target networks, if this is an update step
            if not self.td3 or steps % self.actor_delay == 0:
                self.actor.train(states, np.array(policy_gradient).reshape((-1, self.act_dim)))
                self.actor.train_target()
                self.train_critic_target()

            # Update current state for next step
            old_state = new_state
            steps += 1

        return steps

    def q_value(self, new_states, actor_intent):
        estimate_1 = np.asarray(self.critics[0].target_predict(new_states, actor_intent))
        if not self.td3:
            return estimate_1
        else:
            estimate_2 = np.asarray(self.critics[1].target_predict(new_states, actor_intent))
            return np.min(np.vstack([estimate_1.transpose(), estimate_2.transpose()]), axis=0)

    def gradients(self, states, actions):
        return self.critics[0].gradients(states, actions)

    def train_critic(self, states, actions, critic_target):
        for critic in self.critics:
            critic.train(states, actions, critic_target)

    def train_critic_target(self):
        for critic in self.critics:
            critic.train_target()

    def run_validation_episode(self):
        """Tests the target policy against the environment."""

        steps, total_reward, finished = 0, 0, False
        state = get_state_from(self.environment.reset())

        while not finished and steps < self.step_limit:
            action = self.actor.act_target(state)
            observation, _, finished, _ = self.environment.step(action)
            state = get_state_from(observation)
            total_reward += calculate_reward(observation)
            steps += 1

        return total_reward, steps

    def bellman(self, rewards, q_values, finisheds):
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            future = self.gamma * q_values[i] if not finisheds[i] else 0
            critic_target[i] = rewards[i] + future
        return critic_target

    def sample_batch(self):
        return self.buffer.sample_batch(self.batch_size)

    def save_results(self, performance, duration, steps, validation_steps, episodes):
        """Saves the results of this experiment, i.e. the model weights, config, and outcome."""

        results = open(self.save_path + "_results.txt", "x")
        results.write("***** Results:")
        results.write(f"\nAverage reward: {render_int(np.mean(performance))}")
        results.write(f"\nMaximum reward: {render_int(np.max(performance))}")
        results.write(f"\nTime taken: {render_time(duration)}")
        results.write(f"\nTime finished: {datetime_now()}")
        results.write(f"\nPer-episode lap limit: {render_int(self.lap_limit)}")
        results.write(f"\nPer-episode step limit: {render_int(self.step_limit)}")

        if self.train:
            av_val_steps = validation_steps / episodes * self.episodes_per_validation
            results.write("\n\n\n")
            results.write("***** Training Details:")
            results.write(f"\nUsing TD3: {self.td3}")
            if self.td3:
                results.write(f"\nPolicy update delay: {self.actor_delay}")
            results.write(f"\nUsing action noise: {self.action_noise}")
            results.write(f"\nUsing target policy smoothing: {self.target_policy_smoothing}")
            if self.target_policy_smoothing:
                results.write(f"\nNoise clip: {self.noise_clip}")
            results.write(f"\nUsing parameter noise: {self.param_noise}")
            results.write(f"\nTraining steps: {render_int(steps)}")
            results.write(f"\nTotal episodes: {render_int(episodes)}")
            results.write(f"\nAverage steps per second: {render_int(steps / duration)}")
            results.write(f"\nAverage steps per episode: {render_int(steps / episodes)}")
            results.write(f"\nAverage steps per validation: {render_int(av_val_steps)}")

            results.write("\n\n\n")
            results.write("***** Hyperparameters:")
            results.write("\nActor learning rate: " + str(self.actor_lr))
            results.write("\nCritic learning rate: " + str(self.critic_lr))
            results.write("\nTau: " + str(self.tau))
            results.write("\nSigma: " + str(self.sigma))

        results.write("\n\n\n")
        results.write("***** Config:")
        results.write(f"\nTraining mode: {self.train}")
        results.write(f"\nTrack: {self.track_name}")
        results.write(f"\nEpisodes per validation: {render_int(self.episodes_per_validation)}")
        results.write(f"\nBuffer size: {render_int(self.buffer_size)}")
        results.write(f"\nBatch size: {render_int(self.batch_size)}")

        results.write("\n\n\n")
        results.write("***** Actor:")
        results.write(summarize_model(self.actor.model))
        results.write("\n\n\n")
        results.write("***** Actor Target:")
        results.write(summarize_model(self.actor.target_model))
        results.write("\n\n\n")
        results.write("***** Critic:")
        results.write(summarize_model(self.critics[0].model))
        results.write("\n\n\n")
        results.write("***** Critic Target:")
        results.write(summarize_model(self.critics[0].target_model))
        results.close()

        self.actor.save(self.save_path)
        for i, critic in enumerate(self.critics, start=1):
            critic.save(self.save_path + str(i))

        self.create_plot(performance)

        print("Saved results to \"{}\" successfully.".format(self.save_as))

    def create_plot(self, performance):
        plt.figure()
        plt.title("Performance")
        plt.plot(performance, label='Performance')
        plt.xlabel("Trial")
        plt.ylabel("Performance")
        plt.legend()
        plt.savefig(self.save_path + "_performance.png")


####################################################################################################


def summarize_model(model):
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


def calculate_reward(dict_obs):
    """Calculates the reward signal from a set of state observations."""
    speed = dict_obs["speedX"] * 300
    angle = np.abs(dict_obs["angle"] * np.pi)
    return speed * np.cos(angle) - speed * np.sin(angle)


def get_state_from(dict_obs):
    """
    Takes the observations we received from the `gym` wrapper and combines them into an array of the
    shape expected by our agent.
    """
    return np.hstack((
        dict_obs["angle"],
        dict_obs["speedX"],
        dict_obs["speedY"],
        dict_obs["speedZ"],
        dict_obs["track"],
        dict_obs["trackPos"],
        dict_obs["wheelSpinVel"],
    ))
