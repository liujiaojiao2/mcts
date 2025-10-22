#MDP接口定义
class MDP:
    """ Return all states of this MDP """

    def get_states(self):
        raise NotImplementedError

    """ Return all actions with non-zero probability from this state """

    def get_actions(self, state):
        raise NotImplementedError

    """ Return all non-zero probability transitions for this action
        from this state, as a list of (state, probability) pairs
    """

    def get_transitions(self, state, action):
        raise NotImplementedError

    """ Return the reward for transitioning from state to
        nextState via action
    """

    def get_reward(self, state, action, next_state,path_history):
        raise NotImplementedError

    """ Return true if and only if state is a terminal state of this MDP """

    def is_terminal(self, state):
        raise NotImplementedError

    """ Return the discount factor for this MDP """

    def get_discount_factor(self):
        raise NotImplementedError

    """ Return the initial state of this MDP """

    def get_initial_state(self):
        raise NotImplemented

    """ Return all goal states of this MDP """

    def get_goal_states(self):
        raise NotImplementedError