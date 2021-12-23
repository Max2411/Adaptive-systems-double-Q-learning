import gym
from double_deep_Q_learning.FunctieApproximator import FunctieApproximator

episodes = 10
score_list = []
env = gym.make("LunarLander-v2")    # Create environment
f_approximator = FunctieApproximator()
f_approximator.load_network("deep_q_model_92.pth")

for i in range(episodes):
    observation = env.reset()
    done = False
    while not done:
        env.render()      # show the training
        action = f_approximator.select_action(observation)
        observation, _, done, _ = env.step(action)
env.close()
