from __future__ import  absolute_import

from mcts.stateNode import StateNode
from mcts.mcts import MCTS
# class SingleAgentNode(Node):
#     #实现单智能体MCTS的具体节点逻辑
#     def __init__(self, mdp, parent, state, qfunction, bandit, reward=0.0, action=None):
#         super().__init__(mdp, parent, state, qfunction, bandit, reward, action)
#
#         self.children = {}  # 子节点：{动作: [(子节点, 概率)]}
#
#     def is_fully_expanded(self):
#         # 判断节点是否完全扩展（所有动作均已生成子节点）
#         valid_actions = self.mdp.get_actions(self.state)
#         return len(valid_actions) == len(self.children)
#
#     def select(self):
#         if self.is_fully_expanded() and not self.mdp.is_terminal(self.state):
#             # 完全扩展时用bandit选择动作
#             actions = list(self.children.keys())
#             action = self.bandit.select(self.state, actions, self.qfunction)
#             return self.get_outcome_child(action).select()
#         else:
#             # 未完全扩展时直接返回当前节点
#             return self
#
#     def expand(self):
#         if not self.mdp.is_terminal(self.state):
#             # 随机选择未扩展的动作
#             actions = set(self.mdp.get_actions(self.state)) - set(self.children.keys())
#             if actions:
#                 action = random.choice(list(actions))
#                 self.children[action] = []
#                 return self.get_outcome_child(action)
#         return self  # 终止状态不扩展
#
#     def back_propagate(self, reward, child):
#         # 更新访问次数和Q值，并向父节点传播
#         action = child.action
#         Node.visits[self.state] += 1
#         Node.visits[(self.state, action)] += 1
#
#         delta = (reward - self.qfunction.get_q_value(self.state, action)) / Node.visits[(self.state, action)]
#         self.qfunction.update(self.state, action, delta)
#         if self.parent:
#             # 父节点奖励为当前节点奖励 + 子节点奖励（考虑折扣）
#             parent_reward = self.reward + self.mdp.get_discount_factor() * reward
#             self.parent.back_propagate(parent_reward, self)
#
#     def get_outcome_child(self, action):
#         # 根据动作和转移概率生成子节点
#         next_state, reward, done = self.mdp.execute(self.state, action)
#
#         # 查找已存在的子节点，或创建新节点
#         for child, _ in self.children.get(action, []):
#             if child.state == next_state:
#                 return child
#         new_child = SingleAgentNode(
#             self.mdp, self, next_state, self.qfunction, self.bandit, reward, action
#         )
#         # 记录转移概率（仅用于可视化）
#         transitions = self.mdp.get_transitions(self.state, action)
#         prob = next((p for s, p in transitions if s == next_state), 0.0)
#         self.children.setdefault(action, []).append((new_child, prob))
#         return new_child
#
class SingleAgentMCTS(MCTS):
    #创建根节点实例
    def create_root_node(self,current_state):
        # 创建根节点（初始状态）
        return StateNode(
            self.mdp, None, current_state, self.qfunction, self.bandit
        )




