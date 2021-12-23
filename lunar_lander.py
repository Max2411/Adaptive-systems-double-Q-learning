import gym
from double_deep_Q_learning.FunctieApproximator import FunctieApproximator
import numpy as np
import matplotlib.pyplot as plt


def deep_q_learning() -> None:
    """Start the reinforcement learning algorithm"""
    episodes = 1000
    score_list = []
    env = gym.make("LunarLander-v2")    # Create environment
    f_approximator = FunctieApproximator()

    for i in range(episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            # env.render()      # show the training
            action = f_approximator.select_action(observation)
            new_observation, reward, done, info = env.step(action) # Step in the environment
            f_approximator.train(observation, action, reward, new_observation, done)  # Step in training
            observation = new_observation
            score += reward

        # f_approximator.decay()
        score_list.append(score)
        if (i+1) % 100 == 0:     # Prints average 100 scores every 100 episodes
            print(f"\rEpisode {i+1}/{episodes} finished. Average score: {np.average(score_list[-100:])}", end="")

    f_approximator.save_network(round(np.average(score_list[-100:])))
    env.close()
    plot(score_list)


def plot(scores):
    plt.style.use('fivethirtyeight')
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.show()


if __name__ == "__main__":
    deep_q_learning()
