"""The neural network"""
import torch
"""
Schrijf een function approximator class. Dit is een neuraal netwerk. Gebruik hiervoor een library naar keuze. De agent heeft twee instanties van approximators, een policy-network en een target-network. Begin met een Adam Optimizer met een learning rate van 0.001, RMS loss, en 2 hidden layers met 32 neuronen. De class heeft de volgende functionaliteit:
q-values teruggeven aan de hand van een state of lijst van states
netwerk opslaan
netwerk laden
netwerk trainen
weights handmatig zetten (pas belangrijk bij stap 10)
"""


class FunctieApproximator:
    def __int__(self, learning_rate: float = 0.005) -> None:
        self.lr = learning_rate
        pass
