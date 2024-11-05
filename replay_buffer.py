from collections import deque
import random
import numpy as np

class Buffer():
    def __init__(self, max_len=100_000):
        self.buffer = deque(maxlen=max_len)

    def add(self, element):
        self.buffer.append(element)

    def sample(self, batch_size):
        weights = np.linspace(0, 1, len(self.buffer))

        return random.choices(self.buffer, weights, k=min(batch_size, len(self.buffer)))