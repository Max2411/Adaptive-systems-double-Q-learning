import gym
import torch
from pathlib import Path
from double_deep_Q_learning.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from double_deep_Q_learning.FunctieApproximator import FunctieApproximator
import numpy as np


def deep_q_learing():
    episodes = 1000
    score_list = []
    env = gym.make("LunarLander-v2")
    # settings = (alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env, # TODO kan weg
    #                   batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2,)??
    # greedy_policy = EpsilonGreedyPolicy(env, 0.9) # todo miss weg
    # f_approximator = greedy_policy.functionapproximator
    f_approximator = FunctieApproximator()
    for i in range(episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            # env.render()      # show the training
            # action = env.action_space.sample()
            action = f_approximator.select_action(observation)
            new_observation, reward, done, info = env.step(action)

            f_approximator.train_model(observation, action, reward, new_observation, done)  # TODO

            observation = new_observation
            score += reward
            if done:
                new_observation = env.reset()
        score_list.append(score)
    f_approximator.save_network()
    print(score_list)
    env.close()


if __name__ == "__main__":
    deep_q_learing()
