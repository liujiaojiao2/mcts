import math
import time
import random
from collections import defaultdict

class Node:
    #抽象节点基类，定义基本属性和方法，类比砖块
    visits = defaultdict(lambda: 0)  # 记录状态访问次数
    #visits=0
    next_node_id = 0  # 节点ID计数器
    #  节点类，保存孩子节点集合、父节点以及父节点动作、被访问次数
    def __init__(self, mdp, parent, state, qfunction, bandit, reward=0.0, action=None):
        self.mdp = mdp
        self.parent = parent
        self.state = state
        self.id = Node.next_node_id
        Node.next_node_id += 1
        self.qfunction = qfunction
        self.bandit = bandit
        self.reward = reward
        self.action = action  # 生成该节点的动作
        self.value=0

    def select(self):
        raise NotImplementedError  # 选择节点（需子类实现）

    def expand(self):
        raise NotImplementedError  # 扩展节点（需子类实现）

    def back_propagate(self, reward, child):
        raise NotImplementedError  # 反向传播（需子类实现）

    def get_value(self):
        # 获取当前状态的最大Q值
        #max_q_value = self.qfunction.get_best_q(self.state, self.mdp.get_actions(self.state))
        max_q_value=self.qfunction.get_best_q(self.mdp, self.state)
        return max_q_value

    def get_visits(self):
          visits = Node.visits[self.state]
          return visits
    # def get_child_node(self,best_action,next_state):
    #     """
    #     根据选择的动作返回新的状态节点
    #     """
    #     raise NotImplementedError
class MCTS:
    #  MCTS类，提供算法框架，类比工人，
    def __init__(self, mdp, qfunction, bandit):
        self.mdp = mdp
        self.qfunction = qfunction
        self.bandit = bandit

    # def mcts(self, timeout=1, root_node=None):
    #     if root_node is None:
    #         root_node = self.create_root_node()  # 创建根节点
    #     start_time = time.time()
    #     current_time = time.time()
    #     while current_time < start_time + timeout:
    #         selected_node = root_node.select()  # 选择节点
    #         if not self.mdp.is_terminal(selected_node.state):
    #             child = selected_node.expand()  # 扩展节点。（此处扩展的节点始终是新的状态节点）
    #             reward = self.simulate(child)  # 模拟至终止状态，始终从一个新的状态节点开始模拟
    #             selected_node.back_propagate(reward, child)  # 反向传播
    #         current_time = time.time()
    #     return root_node
    def mcts(self, num_circle=1000, root_node=None):
        if root_node is None:
            root_node = self.create_root_node()  # 创建根节点
        for _ in range(num_circle):
            selected_node = root_node.select()  # 选择节点
            if not self.mdp.is_terminal(selected_node.state):
                child = selected_node.expand()  # 扩展节点。（此处扩展的节点始终是新的状态节点）
                reward = self.simulate(child)  # 模拟至终止状态，始终从一个新的状态节点开始模拟
                selected_node.back_propagate(reward, child)  # 反向传播
        return root_node

    def create_root_node(self,current_state):
        raise NotImplementedError  # 创建根节点（需子类实现）

    def choose(self, state):
        # 随机选择动作（可替换为启发式策略），进行模拟
        return random.choice(self.mdp.get_actions(state))

    def simulate(self, node):
        # 从动作节点开始，随机选择一个状态节点，模拟至终止状态，计算累积折扣奖励
        state = node.state
        done=False
        cumulative_reward = 0.0
        depth = 0
        path_history=[]
        while not done:
            if self.mdp.is_terminal(state):
                #print(f"我进来了哈哈哈哈哈哈哈哈哈哈哈")
                _, reward, done = self.mdp.execute(state, self.mdp.TERMINATE,path_history)
                path_history.append(state)
                cumulative_reward += (self.mdp.get_discount_factor() ** depth) * reward
                break  # 退出循环

            action = self.choose(state)
            (next_state, reward, done )= self.mdp.execute(state, action,path_history)
            path_history.append((state,action))

            cumulative_reward += (self.mdp.get_discount_factor() ** depth) * reward
            depth += 1
            state = next_state

        return cumulative_reward




