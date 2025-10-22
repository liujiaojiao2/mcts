import random


from mcts.mcts import Node
class StateNode(Node):
    """
    表示处于某个状态,可进行决策
    """
    node_type="state"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = {}  # {action:【（env_node，prob），（env_node，prob）】}

    def is_fully_expanded(self):
        return len(self.children) == len(self.mdp.get_actions(self.state))

    def get_next_root_node(self, action, actual_next_state):
        """
        根据执行的动作和实际达到的下一个状态，查找对应的子节点以复用子树。
        :param action: 已执行的最优动作。
        :param actual_next_state: 执行动作后，环境返回的真实下一个状态。
        :return: 对应下一个状态的StateNode子节点，如果找不到则返回None。
        """
        if action not in self.children:
            # 如果该动作不在子节点中，无法复用
            return None

        # 1. 根据action找到对应的环境节点 (EnvironmentNode)
        # 根据您的设计，每个action对应一个env_node
        #env_node = self.children[action][0][0]
        env_node_list = self.children[action]

        # 2. 在环境节点的众多可能结果中，寻找与真实新状态匹配的那个状态节点
        # for state_node, prob in env_node_list:
        #     if state_node.state == actual_next_state:
        #         # 找到了！这个state_node就是我们下一轮MCTS的根节点
        #         state_node.parent = None  # 把它变成根节点，断开与旧父节点的连接
        #         return state_node
        for env_node,prob in env_node_list:
            print(f"当前环境节点的转移概率状态节点为：{env_node.outcomes}")
            for state_node,_ in env_node.outcomes:
                if state_node.state == actual_next_state:
                    # 找到了！这个state_node就是我们下一轮MCTS的根节点
                    state_node.parent = None  # 把它变成根节点，断开与旧父节点的连接
                    return state_node

        # 如果因为某种原因没找到（理论上不应该发生，除非有bug），返回None
        return None
    def select(self):
        if self.is_fully_expanded():
            #actions = list(self.children.keys())
            action = self.bandit.select(self.state, self.qfunction, self.children)
            env_nodes=[env_node for env_node,_ in self.children[action]]
            selected_node=random.choices(env_nodes,weights = [prob for _,prob in self.children[action]])[0]# 当前选中的节点为环境节点
            return selected_node.select()
        else:
            return self

    def expand(self):
        from mcts.environmentNode import EnvironmentNode
        if not self.mdp.is_terminal(self.state):
            unexpanded_actions = set(self.mdp.get_actions(self.state)) - set(self.children.keys())
            if unexpanded_actions:
                action = random.choice(list(unexpanded_actions))# 随机选择一个未扩展的动作节点，可优化
                # 1、只创建一个 env_node
                env_node = EnvironmentNode(self.mdp, parent=self, state=self.state,
                                           qfunction=self.qfunction, bandit=self.bandit,
                                           reward=0.0, action=action)
                self.children[action] = [(env_node, 1.0)]  # 概率为1，因为env_node内部再分概率

                #============继续扩展所有的状态节点
                # 2. 立刻让环境节点扩展出下一层的状态节点
                # env_node.expand() 会返回一个用于模拟的 StateNode
                state_node_for_simulation = env_node.expand()

                # 3. 返回这个真正的“新”节点作为模拟的起点
                return state_node_for_simulation

        return self

    def back_propagate(self, reward, child):
        """
        向前传播吗，从新扩展的节点沿着选择的所有节点向前传播
        选所有动作节点中的最大q值为value
        :param reward:模拟中得到的最终回报，这个值在传播过程中保持不变
        :param child:被扩展的节点，从哪个子节点传回的价值
        :return:
        """
        # 更新访问次数和Q值，并向父节点传播
        action=child.action
        Node.visits[self.state] += 1
        Node.visits[(self.state, action)] += 1

        # 计算学习率 (1/访问次数)
        alpha = 1.0 / Node.visits[(self.state, action)]


        # 直接使用 reward 作为 target，让 QTable.update() 处理mcts更新
        self.qfunction.update(self.state, child.action, reward, alpha)

        # 3. 重新计算当前状态的V(s)值 (核心)
        # V(s) = max_a Q(s, a)
        # 假设qfunction有一个方法可以获取一个状态下所有动作的Q值中的最大值
        max_q_value = self.qfunction.get_best_q(self.mdp, self.state)
        self.value = max_q_value

        #向上递归传播
        if self.parent:
            #parent_reward = self.reward + self.mdp.get_discount_factor() * reward
            # 直接将原始的、未经修改的最终回报 reward 传递给父节点。
            # 折扣因子的作用会通过大量更新，最终体现在Q函数学习到的价值中，
            # 而不是在单次反向传播中手动计算。
            self.parent.back_propagate(reward, self)

    def choose_best_action(self):
        """
        根据构建的mcts树，选择最优动作执行到新状态
        选择依据：最大q值对应的动作
        返回：best_action
        """
        if not self.children:
            # 如果没有子节点（未扩展），则无法选择动作
            return None
        best_action = None
        max_visits = -1

        # 遍历所有已探索的动作及其对应的环境节点
        for action, env_node_list in self.children.items():
            # 获取该动作的访问次数
            visits = Node.visits.get((self.state, action), 0)

            if visits > max_visits:
                max_visits = visits
                best_action = action

        return best_action