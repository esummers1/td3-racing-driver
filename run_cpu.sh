#!/usr/bin/env bash

cd src || exit

# This guarantees use of CPU rather than GPU, regardless of which you use for training. This is
# sufficient to provide replicable validation results when combined with setting the three seeds
# (python.random, np.random, tf.random).
CUDA_VISIBLE_DEVICES="" python main.py
