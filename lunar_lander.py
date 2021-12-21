import gym
import torch
from pathlib import Path
from double_deep_Q_learning.EpsilonGreedyPolicy import EpsilonGreedyPolicy
import numpy as np


def dnq():
    episodes = 5
    score_list = []
    env = gym.make("LunarLander-v2")
    # settings = (alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
    #                   batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2,)??
    greedy_policy = EpsilonGreedyPolicy(env, 0.9)
    f_approximator = greedy_policy.functionapproximator
    for i in range(episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            # env.render()      # show the training

            action = greedy_policy.select_action(observation, model=None)  # TODO model parameter is for a test
            # action = env.action_space.sample()
            new_observation, reward, done, info = env.step(action)
            agent.step(observation, action, reward, new_observation, done)  # TODO
            f_approximator.step(observation, action, reward, new_observation, done)  # TODO
            observation = new_observation
            score += reward
            if done:
                new_observation = env.reset()
        score_list.append(score)
    f_approximator.save_network()
    print(score_list)
    env.close()


if __name__ == "__main__":
    dnq()
