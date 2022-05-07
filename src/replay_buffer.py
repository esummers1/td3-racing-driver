import random
import numpy as np

from collections import deque


class ReplayBuffer(object):
    """
    Buffer for experience replay, storing (s, a, r, d, s+1) tuples up to a configured size limit.
    Once the limit is reached, new entries will displace the oldest ones.
    """

    def __init__(self, size_limit):
        self.count = 0
        self.size_limit = size_limit
        self.buffer = deque()

    def add(self, state, action, reward, done, new_state):
        """Adds a new entry to the buffer, discarding the oldest entry if it is already full."""

        experience = (state, action, reward, done, new_state)
        if self.count < self.size_limit:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample_batch(self, batch_size):
        """
        Gets a random sample of the buffer's contents. If the buffer does not contain enough data,
        its entire contents will be used.
        """

        size = min(self.count, batch_size)
        batch = random.sample(self.buffer, size)

        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])
        return s_batch, a_batch, r_batch, d_batch, new_s_batch
