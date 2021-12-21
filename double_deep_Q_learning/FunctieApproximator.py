"""The neural network"""
import torch
from torch import nn            # TODO remove torchvision if unnessassery
from torchvision import models
import random

from typing import List
from pathlib import Path

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
    def __int__(self, learning_rate: float = 0.001, batch_size: int = 10, epsilon: float = 0.92) -> None:
        self.policy_network = DeepQlearning(learning_rate=learning_rate)    # TODO is dit hier nodig
        self.target_network = DeepQlearning(learning_rate=learning_rate)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss

        self.lr = learning_rate
        self.action_space = [0, 1, 2, 3]
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.memory = Memory(size=10000, batch_size=batch_size)

    def get_qvalues(self, states: List[None]) -> None:  # TODO: Typing State?
        pass                                            # TODO Nodig??

    def save_network(self) -> None:
        model_name = "deep_q_model"
        torch.save(Path("models") / model_name)

    def load_network(self, model_name: str):    # TODO
        model = torch.load(Path("models") / model_name)
        model.eval()
        return model

    def select_action(self, state) -> int:      # TODO

        if random.random() > self.epsilon:
            state = torch.tensor([state])
            actions = self.policy_network.forward(state)
            action = torch.argmax(actions).item()     # TODO
        else:
            action = random.choices(self.action_list)[0]
        return action

    def decay(self):    # TODO optional
        pass

    def train_model(self, state, action, reward, next_state, done) -> None:     # TODO
        self.memory.record(state, action, reward, next_state, done)
        sample = self.memory.sample(batchsize=self.batch_size)
        state = torch.tensor(state)
        state.unsqueeze(0)


    def set_weights(self) -> None:
        pass

    def load_weights(self) -> None:
        pass

