class Agent:
    def __int__(self, policy_network, target_network) -> None:
        self.policy_network = policy_network
        self.target_network = target_network
        # TODO: Gamma/Alpha/Batchsize/Epsilon

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
