"""The neural network"""
import torch
from torch import nn  # TODO remove torchvision if unnessassery
import torch.nn.functional as F
from torchvision import models
import random

from typing import List
from pathlib import Path
import os

from .Memory import Memory
from .DeepQlearning import DeepQlearning

"""
Schrijf een function approximator class. Dit is een neuraal netwerk. Gebruik hiervoor een library naar keuze. De agent heeft twee instanties van approximators, een policy-network en een target-network. Begin met een Adam Optimizer met een learning rate van 0.001, RMS loss, en 2 hidden layers met 32 neuronen. De class heeft de volgende functionaliteit:
q-values teruggeven aan de hand van een state of lijst van states
netwerk opslaan
netwerk laden
netwerk trainen
weights handmatig zetten (pas belangrijk bij stap 10)
"""


class FunctieApproximator:
    def __init__(self, learning_rate: float = 0.001, batch_size: int = 10, epsilon: float = 0.92, gamma: float = 0.99,
                 update_interval: int = 10, tau: float = 0.3) -> None:
        self.policy_network = DeepQlearning()  # TODO is dit hier nodig
        self.target_network = DeepQlearning()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss
        self.lr = learning_rate

        self.action_space = [0, 1, 2, 3]
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.update_interval = update_interval
        self.counter = 0
        self.max_mem = 0
        self.mem_size = 10000

        self.memory = Memory(size=self.mem_size, batch_size=batch_size)

    def save_network(self) -> None:
        model_name = "deep_q_model"
        # torch.save(os.path.join(f"models", model_name))
        torch.save(self.policy_network, Path("models") / model_name)

    def load_network(self, model_name: str):  # TODO
        model = torch.load(Path("models") / model_name)
        model.eval()
        return model

    def select_action(self, state) -> int:  # TODO
        if random.random() > self.epsilon:
            state = torch.tensor([state])
            actions = self.policy_network.forward(state)
            action = torch.argmax(actions).item()  # TODO
        else:
            action = random.choices(self.action_space)[0]
        return action

    def decay(self):  # TODO optional
        pass

    def train_model(self, state, action, reward, next_state, done) -> None:  # TODO + maybe new function name
        self.memory.record(state, action, reward, next_state, done)
        self.max_mem += 1 if self.max_mem < self.mem_size else self.mem_size

        self.counter += 1
        self.counter %= self.update_interval
        if self.counter == 0:
            if self.memory.__len__() > self.batch_size:
                sample = self.memory.sample()
                self.calc_q(sample)

    def calc_q(self, sample):
        states, actions, rewards, next_states, done = sample

        q_targets_next = self.target_network(next_states).max(1)[0].unsqueeze(1)

        q_target = rewards + self.gamma * q_targets_next * (1 - done)

        expected_q = self.policy_network(states).gather(1, actions)

        loss = F.mse_loss(expected_q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO: Update target network
        self.copy_network()

    def copy_network(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def set_weights(self) -> None:
        pass

    def load_weights(self) -> None:
        pass
