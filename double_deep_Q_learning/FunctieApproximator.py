"""The neural network"""
import torch
from torch import nn  # TODO remove torchvision if unnessassery
import torch.nn.functional as F

import random
from pathlib import Path
import numpy as np

from .Memory import Memory
from .DeepQlearning import DeepQlearning


class FunctieApproximator:
    def __init__(self, learning_rate: float = 5e-4, batch_size: int = 64, epsilon: float = 0.92, epsilon_decay: float = 0.995,
                 min_epsilon: float = 0.01, gamma: float = 0.99, update_interval: int = 4, tau: float = 1e-3) -> None:
        # Neural Networks
        self.policy_network = DeepQlearning()
        self.target_network = DeepQlearning()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss
        # Hyperparameters
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.tau = tau
        # Neural Network updates
        self.update_interval = update_interval
        self.counter = 0

        # Memory
        self.mem_size = 10000
        self.memory = Memory(size=self.mem_size, batch_size=batch_size)

        self.action_space = [0, 1, 2, 3]

    def save_network(self, score) -> None:
        """Save the policy model to the models folder."""
        model_name = f"deep_q_model_{score}"
        torch.save(self.policy_network, Path("models") / model_name)

    def load_network(self, model_name: str) -> None:
        """Load a policy out of the models folder"""
        model = torch.load(Path("models") / model_name)
        model.eval()
        return model

    def select_action(self, state) -> int:
        """Selects an action based of the policy according to epsilon else select a random action."""
        if random.random() < self.epsilon:
            self.policy_network.eval()          # Sets policy network to eval mode to optain actions
            state = torch.tensor(np.asarray([state]))       # Converts state to tensor
            actions = self.policy_network.forward(state)    # Optians the values of each actions
            self.policy_network.train()         # Sets policy network back to training mode
            action = torch.argmax(actions).item()   # Picks the best actions
        else:
            action = random.choices(self.action_space)[0]   # Picks a random action out of the action_space
        return action

    def decay(self) -> None:
        """
        Decreases the epsilon until the minimum epsilon is reached
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon

    def train(self, state, action, reward, next_state, done) -> None:
        """
        Start training the model after the update_interval amount of iterations. While making sure there is enough
        data to train
        """
        self.memory.record(state, action, reward, next_state, done)

        self.counter += 1
        self.counter %= self.update_interval
        if self.counter == 0:       # Prevents from updating every step
            if self.memory.__len__() > self.batch_size:     # Check if there is enough data for a batch
                sample = self.memory.sample()
                self.double_deep_q(sample)

    def double_deep_q(self, sample) -> None:
        """
        Double Deep Q-Learning Algorithm and loss calculation.
        """
        states, actions, rewards, next_states, done = sample

        next_q_target = self.target_network(next_states).max(1)[0].unsqueeze(1)    # Takes the best action for the next state
        target_q_values = rewards + self.gamma * next_q_target * (1 - done)   # Bellman equation
        expected_q = self.policy_network(states).gather(1, actions)

        loss = F.mse_loss(expected_q, target_q_values)  # Calculates the Mean Squared Error
        self.optimizer.zero_grad()
        loss.backward()     # Applies backpropagation with the MSE loss
        self.optimizer.step()
        # Update target_network
        self.copy_network()

    def copy_network(self):
        """Replaces part of the target network work with the policy network according to given tau."""
        for target_param, policy_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
