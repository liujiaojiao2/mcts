from fontTools.ttLib.ttVisitor import visit

from multi_armed_bandit.MultiArmedBandit import MultiArmedBandit
from collections import defaultdict
import math
import random
from mcts.mcts import Node
class UpperConfidenceBounds(MultiArmedBandit):
    """
    Upper Confidence Bound bandit
    """
    def __init__(self,c=math.sqrt(1)):
        #self.total = 0
        self.c=c
        # number of times each action has been chosen
        #self.times_selected = {}

    def select(self, state,  qfunction,children):
        actions = list(children.keys())
        times_selected={}
        total=0
        # First execute each action one time
        for action in actions:

            #visits = sum(child.get_visits() for child, _ in children[action])
            visits=Node.visits.get((state, action),0)
            times_selected[action] = visits
            total += visits
        # --- UCB1 核心逻辑 ---

        # 优先选择从未被探索过的动作
        for action in actions:
            if times_selected[action] == 0:
                return action
        #UCB选择
        # 如果所有动作都至少被探索过一次，则使用UCB1公式
        max_actions = []
        max_value = float("-inf")

        for action in actions:
            #UCB=Q_value+探索值，
            value = qfunction.get_q_value(state, action) + self.c*math.sqrt(
                (math.log(total)) / times_selected[action]
            )

            if value > max_value:
                max_actions = [action]
                max_value = value
            elif value == max_value:
                max_actions += [action]

        return random.choice(max_actions)
    def reset(self):
        self.total=0
        self.times_selected.clear()