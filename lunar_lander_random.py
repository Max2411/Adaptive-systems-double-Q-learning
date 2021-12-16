import gym
import time

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    observation = env.reset()

    for _ in range(1000):
        env.render()
        # action_space = gym.spaces.Discrete(1)
        action_space = env.action_space.n
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()
