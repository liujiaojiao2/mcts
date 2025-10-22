class MultiArmedBandit():
    """ Select an action for this state given from a list given a Q-function """

    def select(self, state, actions, qfunction):
        raise NotImplementedError

    """ Reset a multi-armed bandit to its initial configuration """

    def reset(self):
        self.__init__()