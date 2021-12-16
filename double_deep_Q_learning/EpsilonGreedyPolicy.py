import random

import gym


class EpsilonGreedyPolicy:
    def __init__(self, env, epsilon):#, model):
        self.env = env
        self.action_space = env.action_space.n
        self.action_list = [i for i in range(self.action_space)]
        # self.model = model
        self.epsilon = epsilon

    def select_action(self, state, model) -> int:
        state = state
        action = random.choices(self.action_list)
        return action
