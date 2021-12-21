from Memory import Memory

    # TODO Remove file if unnesessary

class Agent:
    def __int__(self, policy_network, target_network, gamma, epsilon) -> None:
        self.policy_network = policy_network
        self.target_network = target_network
        # TODO: Gamma/Alpha/Batchsize/Epsilon
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = Memory(size=10000)

    def train(self) -> None:
        """
        Update the policy-network with Double Deep Q-Learning algorithm
        :return:
        """
        pass

    def copy_model(self, tao: float) -> None:
        """
        Combine the policy- and target-networks.
        :param tao: Percentage of target-network that gets replaced by policy-network
        :return:
        """
        pass
