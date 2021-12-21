import random
import torch
import gym
from .FunctieApproximator import FunctieApproximator
import numpy as np


class EpsilonGreedyPolicy:
    def __init__(self, env, epsilon: float = 0.93) -> None:
        self.env = env
        self.action_space = env.action_space.n
        self.action_list = [i for i in range(self.action_space)]

        self.functionapproximator = FunctieApproximator()
        self.network = self.functionapproximator.policy_network
        self.epsilon = epsilon

    def select_action(self, state, model) -> int:

        if random.random() > self.epsilon:
            state = torch.tensor([state])
            actions = self.network.forward(state)
            action = torch.argmax(actions).item()     # TODO
        else:
            action = random.choices(self.action_list)[0]
        return action

    def decay(self):    # TODO optional
        pass
