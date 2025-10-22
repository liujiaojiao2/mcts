from collections import defaultdict


class QPolicy:
    def get_action(self,state,actions):
        raise NotImplementedError
class DeterministicPolicy(QPolicy):
    """
    实现确定策略,表格形式实现

    """
    def __init__(self, default_action=None):
        self.policy_table = defaultdict(lambda: default_action)
    def update(self, state, action):
        self.policy_table[state] = action
    def get_action(self, state,actions):
        return self.policy_table[state]

class StochasticPolicy(QPolicy):
    #随机策略，可以但没必要
    def __init__(self):
        self.policy_table = defaultdict(dict)   #状态动作对概率表 {state:{action:prob}}
    def update(self, states, actions, rewards):
        pass
    def select_action(self,state,actions):
        pass
    def get_probability(self, state, action):
        pass
