from mcts.mcts import Node


import random
class  EnvironmentNode(Node):
    """
    表示环境对动作的响应，包含多个可能的状态（概率）
    """
    node_type="env"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = []  # List of state_node（state_node，prob）
        self.children = {}  #{action：【（child_node，prob），（）……【}

    def select(self):
        # 从该节点选择子节点（状态）
        if self.outcomes:
            # 根据概率加权选择子节点
            outcomes, probs = zip(*self.outcomes)
            selected_outcome = random.choices(outcomes, weights=probs, k=1)[0]
            return selected_outcome.select()
        else:
            return self

    def expand(self):
        from mcts.stateNode import StateNode
        transitions = self.mdp.get_transitions(self.state, self.action)

        for next_state, prob in transitions:
            if not any(child.state == next_state for child,_ in self.outcomes):
                new_state_node = StateNode(
                    self.mdp, parent=self, state=next_state,
                    qfunction=self.qfunction, bandit=self.bandit,
                    reward=0.0,  # 初始奖励为 0，实际奖励在回溯时更新
                    action=self.action
                )
                self.outcomes.append((new_state_node, prob))  # 记录子节点及其概率
                 # 每次扩展只添加一个新节点（符合 MCTS 的标准行为）
            
        self.children[self.action] = self.outcomes  # 更新 children 字典
        outcomes, probs = zip(*self.outcomes)
        return random.choices(outcomes, weights=probs, k=1)[0]
    def back_propagate(self, reward, child):
        # 传播到父节点（StateNode），维护一个q值表，更新动作节点的reward
        # 加权计算所有状态节点的平均值，how：循环计算每一个or直接计算全部
        # 1. 重新计算期望价值 (核心)
        # 遍历所有已知的子节点（StateNode），计算加权平均值
        expected_value = 0.0
        if self.outcomes:
            for state_node, prob in self.outcomes:
                # 假设每个Node都有一个 .value 属性
                expected_value += prob * state_node.value
        self.value = expected_value

        # 2. 递归调用父节点
        if self.parent:
            self.parent.back_propagate(reward, self)
