import os
import random

import gym

import numpy as np
import tensorflow as tf

from experiment import Experiment
from file_utils import get_track_name, select_track


def main():
    set_random_seeds()
    reduce_log_noise()
    setup_tensorflow()

    experiments = [
        observe(load_from="v53_a", track="road/wheel-2"),
    ]
    for index, experiment in enumerate(experiments):
        original_track = get_track_name()
        selected_track = experiment.track_name

        try:
            print("\n\n\n**** INFO| Starting experiment {}. |****".format(index))
            select_track(original_track, selected_track)
            experiment.port = index
            experiment.run()
            print("**** INFO| Finished experiment {}. |****".format(index))
        except Exception as e:
            print("**** ERROR|"
                  " Experiment {} failed with the following error: `{}` "
                  "|****".format(index, e))
        finally:
            select_track(selected_track, original_track)


def setup_tensorflow():
    """
    Disables TensorFlow eager execution.

    This is a bit of a hack, since we would ideally like to use it, but this would require some
    rework of the existing code to use tf.GradientTapes. The issue is that tf.gradient is not
    supported with eager execution in TensorFlow 2+.
    """
    tf.compat.v1.disable_eager_execution()
    print("\n**** INFO| GPU Available:", tf.test.is_gpu_available(), " |****")


def reduce_log_noise():
    """Reduces log noise from Gym and TensorFlow."""
    gym.logger.set_level(40)
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def set_random_seeds():
    """Does not guarantee determinism if running on GPU due to multithreading."""
    seed = 83
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def observe(load_from="untitled0", track=get_track_name(), laps=3, steps=1e9):
    """Runs a single validation episode with rendering enabled."""
    return inspect(load_from, track, render=True, laps=laps, steps=steps)


def inspect(load_from="untitled0", track=get_track_name(), render=False, laps=3, steps=1e9):
    """
    Runs a single validation episode with rendering disabled. This is helpful for getting the score
    more quickly.
    """
    return Experiment(
        render=render,
        load_from=load_from,
        track=track,
        lap_limit=laps,
        train=False,
        step_limit=steps)


if __name__ == "__main__":
    main()
