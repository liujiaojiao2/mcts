import graphviz

from mcts.environmentNode import EnvironmentNode
from mcts.single_agent_mcts import SingleAgentMCTS
from mcts.stateNode import StateNode


class  GraphVisualisation:
    def __init__(self, max_level=3):
        self.max_level = max_level

    def single_agent_mcts_to_graph(self, root_node, filename="mcts_tree"):
        """
        :param root_node:
        :param filename:
        :return: graph
        """
        #创建一个有向图
        dot = graphviz.Digraph()
        #递归添加节点和边
        self.add_node(dot,root_node,level=0)

        #保存并返回图
        dot.render(filename,view=True)
        return dot
    def add_node(self,dot,node,level):
        if level>self.max_level:
            return

        #添加当前节点
        if node.node_type == "state":
            node_lable = f"state({node.state})={node.get_value():.2f}\nN={node.get_visits()}"
            dot.node(str(node.id), node_lable, **self.get_node_attribute(node))
        else:
            #node_lable = f"env({node.state})={node.get_value():.2f}\nN={node.get_visits()}"
            dot.node(str(node.id), **self.get_node_attribute(node))

        # 递归添加子节点和边
        if isinstance(node, StateNode):
            for action, env_list in node.children.items():
                for child, probability in env_list:
                    print(env_list)
                    #child_label = f"V2222222({child.state})={child.get_value():.2f}\nN={child.get_visits()}"
                    #dot.node(str(child.id),  **self.get_node_attribute(child))
                    edge_label = f"{action}"
                    dot.edge(str(node.id), str(child.id), label=edge_label)
                    self.add_node(dot, child, level + 1)
        elif isinstance(node, EnvironmentNode):
            for child, probability in node.outcomes:  # 直接迭代 outcomes
                child_label = f"V({child.state})={child.get_value():.2f}\nN={child.get_visits()}"
                dot.node(str(child.id), child_label, **self.get_node_attribute(child))
                edge_label = f"{node.action}:{probability:.2f}"  # 使用 node.action 作为动作标签
                dot.edge(str(node.id), str(child.id), label=edge_label)
                self.add_node(dot, child, level + 1)
    def get_node_attribute(self,node):
        if isinstance(node,StateNode):
            return {
                'shape':'circle',
                'style':'filled',
                'fillcolor':'white',
                'color':'black'
            }
        elif isinstance(node,EnvironmentNode):
            return {
                'shape':'box',
                'style':'filled',
                'fillcolor':'black',
                'color':'black',
                #'fontcolor':'white'
            }
        else:
            return {
                'shape':'circle',
                'style':'filled',
                'fillcolor':'lightgray',
                'color':'black'
            }
if __name__  == '__main__':
    from mdp.gridworld import GridWorld
    from graph_visualisation import GraphVisualisation
    from q_learning.q_table import QTable
    from mcts.single_agent_mcts import SingleAgentMCTS
    from q_learning.q_policy import DeterministicPolicy
    from multi_armed_bandit.ucb import UpperConfidenceBounds

    # 创建网格世界
    gridworld = GridWorld()
    # 创建Q值表
    qfunction = QTable()
    # 创建MCTS并进行搜索
    root_node = SingleAgentMCTS(gridworld, qfunction, UpperConfidenceBounds()).mcts(timeout=0.03)
    # 可视化树
    gv = GraphVisualisation(max_level=6)
    graph = gv.single_agent_mcts_to_graph(root_node, filename="mcts_tree")
    graph