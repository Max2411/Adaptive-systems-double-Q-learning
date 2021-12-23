import collections
import random
import numpy as np
import torch
from typing import Tuple


class Memory:
    def __init__(self, size, batch_size) -> None:
        self.size = size
        self.deque = collections.deque(maxlen=size)
        self.batch_size = batch_size
        self.exp = collections.namedtuple("experience", ["state", "action", "reward", "next_state", "done"])

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gets sample out of deque and convert the data to tensor for training purpusses."""
        sample = random.sample(self.deque, self.batch_size)
        # Creates seperate list for each point in the experience
        states = []
        actions = []
        rewards = []
        next_states = []
        done = []
        for s in sample:    # Creates lists of seperate data
            # Appends data to the correct list
            states.append(s.state)
            actions.append([s.action])  # Converts to value to listso that each value will have the same dimensions
            rewards.append([s.reward])
            next_states.append(s.next_state)
            done.append([s.done])
        # Convert data to tensors after converting the lists to arrays
        states = torch.from_numpy(np.asarray(states)).float()
        actions = torch.from_numpy(np.asarray(actions)).long()
        rewards = torch.from_numpy(np.asarray(rewards)).float()
        next_states = torch.from_numpy(np.asarray(next_states)).float()
        done = torch.from_numpy(np.asarray(done).astype(np.uint8))
        return (states, actions, rewards, next_states, done)

    def record(self, state, action, reward, next_state, done) -> None:
        """Append new memory to the deque."""
        mem = self.exp(state, action, reward, next_state, done)
        self.deque.append(mem)

    def __len__(self) -> int:
        """Gives length of the deque."""
        return len(self.deque)
