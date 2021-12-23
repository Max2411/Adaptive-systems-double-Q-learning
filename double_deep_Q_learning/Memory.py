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

    def sample(self):
        sample = random.sample(self.deque, self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        done = []
        for s in sample:
            states.append(s.state)
            actions.append([s.action])
            rewards.append([s.reward])
            next_states.append(s.next_state)
            done.append([s.done])
        states = torch.from_numpy(np.asarray(states)).float()
        actions = torch.from_numpy(np.asarray(actions)).long()
        rewards = torch.from_numpy(np.asarray(rewards)).float()
        next_states = torch.from_numpy(np.asarray(next_states)).float()
        done = torch.from_numpy(np.asarray(done).astype(np.uint8))
        return (states, actions, rewards, next_states, done)

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
