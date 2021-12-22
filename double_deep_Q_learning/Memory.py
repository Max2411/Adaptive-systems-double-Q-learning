import collections
import random
import numpy as np
import torch

class Memory:
    def __init__(self, size, batch_size) -> None:
        self.size = size
        self.deque = collections.deque(maxlen=size)
        self.batch_size = batch_size
        self.exp = collections.namedtuple("experience", ["state", "action", "reward", "next_state", "done"])
        self.state = np.zeros((size, 8), dtype=np.float32)
        self.action = np.zeros(size, dtype=np.int32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.next_state = np.zeros((size,8), dtype=np.float32)
        self.done = np.zeros(size, dtype=bool)
        self.counter = 0

    def sample(self):  # todo return to 'def sample(self):'
        """Returns a sample of the memory"""
        sample = random.sample(self.deque, self.batch_size)
        checked = False
        for exp in sample:
            if not checked:
                state = np.array([exp.state])
                action = np.array([exp.action])
                reward = np.array([exp.reward])
                next_state = np.array([exp.next_state])
                done = np.array(exp.done)
            else:
                state = np.append(state, [exp.state], axis=0)
                action = np.append(action, [exp.action], axis=0)
                reward = np.append(reward, [exp.reward], axis=0)
                next_state = np.append(next_state, [exp.next_state], axis=0)
                done = np.append(done, [exp.done], axis=0)
            checked = True
        return (state, action, reward, next_state, done)


    def record(self, state, action, reward, next_state, done) -> None:
        """Append new memory to the memory list"""
        self.counter %= self.size
        mem = self.exp(state, action, reward, next_state, done)
        self.state[self.counter] = state
        self.action[self.counter] = action
        self.reward[self.counter] = reward
        self.next_state[self.counter] = next_state
        self.done[self.counter] = done
        self.counter += 1
        self.deque.append(mem)

    def __len__(self) -> int:
        return len(self.deque)
