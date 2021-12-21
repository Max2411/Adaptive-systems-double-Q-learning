import collections
import random


class Memory:
    def __init__(self, size, batch_size) -> None:
        self.deque = collections.deque(maxlen=size)
        self.batch_size = batch_size
        self.exp = collections.namedtuple("experience", ["state", "action", "reward", "next_state", "done"])

    def sample(self):
        """Returns a sample of the memory"""
        return random.sample(self.deque, self.batch_size)

    def record(self, state, action, reward, next_state, done) -> None:
        """Append new memory to the memory list"""
        mem = self.exp(state, action, reward, next_state, done)
        self.deque.append(mem)
