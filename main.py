from mdp.gridworld import GridWorld
#from graph_visualisation import GraphVisualisation
from q_learning.q_table import QTable
from mcts.single_agent_mcts import SingleAgentMCTS
from q_learning.q_policy import DeterministicPolicy
from multi_armed_bandit.ucb import UpperConfidenceBounds
import random
import numpy as np
#1. 初始化
#创建网格世界（增加不确定性）
gridworld=GridWorld(noise=0.1)  # 20%的概率会滑向侧边
#创建Q值表
qfunction=QTable()
#创建UpperConfidenceBounds
bandit = UpperConfidenceBounds()
mcts_agent = SingleAgentMCTS(gridworld, qfunction, bandit) # 假设这是您实现了MCTS的类

# 固定随机种子，确保可复现
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# 配置多轮实验参数
num_episodes = 10
steps_list = []
returns_list = []

for episode in range(num_episodes):
    #2、从起点开始（每轮重置环境轨迹与计数）
    gridworld.reset()
    current_state=gridworld.get_initial_state()
    #为起点创建一个MCTS节点
    start_node = mcts_agent.create_root_node(current_state)
    current_node=start_node
    #3、外部循环，直到到达终点
    pace=0
    total_reward=0
    total_discount_reward=0
    path_history=[]
    while not gridworld.is_terminal(current_state):
        print(f"智能体当前位置: {current_state}, 开始思考... (Episode {episode+1}/{num_episodes})")

        #--------思考阶段 ----
        #以当前节点为根，进行3s思考
        #返回一个包含大量节点的树的根节点

        # root_node = mcts_agent.mcts(timeout=10.0, root_node=current_node)
        root_node = mcts_agent.mcts(num_circle=5000, root_node=current_node)
        # ---- 决策阶段 ------
        # 从构建的mcts树中选择最佳动作（访问次数最多or最大q值）
        best_action=(root_node.choose_best_action())
        print(f"智能体选择动作: {best_action}")

        # -------- 执行阶段 ----
        # 在网格世界中执行该动作，得到新状态
        next_state, reward ,done= gridworld.execute(current_state, best_action,path_history)
        path_history.append(current_state)
        print(f"智能体移动到新位置: {next_state}，此时的回报={reward}")
        #total_reward+=reward
        total_discount_reward+=(gridworld.discount_factor** pace)*reward
        pace+=1

        # ----------状态更新与子树复用-------
        current_state=next_state

        #复用子树，不需重新开始构建一棵mcts树
        current_node = root_node.get_next_root_node(best_action, next_state)

    print(f"智能体已到达终点,本轮步长={pace}，本轮总回报={total_discount_reward}")
    steps_list.append(pace)
    returns_list.append(total_discount_reward)

# 统计与输出多轮平均指标
avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0.0
avg_return = sum(returns_list) / len(returns_list) if returns_list else 0.0
print(f"多轮结果：轮数={num_episodes}，平均步长={avg_steps:.2f}，平均总回报={avg_return:.2f}")



#
#
# #可视化树
# gv=GraphVisualisation(max_level=6)
# graph=gv.single_agent_mcts_to_graph(root_node,filename="mcts_tree")
# graph
# #可视化Q值表 - 转换数据格式
# q_table_for_viz = {}
# for ((state, action), q_value) in qfunction.q_table.items():
#     x, y = state
#     q_table_for_viz[(x, y, action)] = q_value
#
# # print("转换后的 Q-table 格式:")
# # for key, value in list(q_table_for_viz.items())[:5]:
# #     print(f"  {key}: {value:.4f}")
#
# gridworld.visualise_q_function(q_table_for_viz)
# # 从Q表中提取策略 - 只为学习过的状态提取策略
# # print("\n--- 提取策略 ---")
#
# # 手动提取策略，只考虑实际学习过的状态
# learned_policy = {}
# learned_states = set()
#
# # 获取所有学习过的状态
# for (state, action), q_value in qfunction.q_table.items():
#     learned_states.add(state)
#
# print(f"学习过的状态数量: {len(learned_states)}")
# print("学习过的状态:", sorted(learned_states))
#
# # 为每个学习过的状态找到最优动作
# for state in learned_states:
#     if gridworld.is_terminal(state):
#         continue
#
#     actions = gridworld.get_actions(state)
#     max_q = float("-inf")
#     best_action = None
#
#     for action in actions:
#         q_value = qfunction.get_q_value(state, action)
#         if q_value > max_q:
#             max_q = q_value
#             best_action = action
#
#     if best_action is not None:
#         learned_policy[state] = best_action
#
# print(f"提取的策略大小: {len(learned_policy)}")
# print("策略详情:")
# for state, action in sorted(learned_policy.items()):
#     q_value = qfunction.get_q_value(state, action)
#     print(f"  State={state}: Action={action} (Q={q_value:.4f})")
#
# # 可视化策略
# gridworld.visualise_policy(learned_policy)
#
# # 验证实际路径执行
# print("\n--- 验证实际路径执行 ---")
# print("使用学习到的策略执行一条完整路径：")
#
# current_state = gridworld.get_initial_state()
# path = [current_state]
# max_steps = 50  # 增加最大步数
# visited_count = {}  # 记录每个状态的访问次数
#
# for step in range(max_steps):
#     if gridworld.is_terminal(current_state):
#         print(f"到达终点！")
#         break
#
#     # 更新访问计数
#     visited_count[current_state] = visited_count.get(current_state, 0) + 1
#
#     # 检测循环：如果一个状态被访问超过3次，尝试不同的策略
#     if visited_count[current_state] > 3:
#         print(f"检测到循环！状态 {current_state} 已访问 {visited_count[current_state]} 次")
#         # 强制使用随机策略打破循环
#         actions = gridworld.get_actions(current_state)
#         if actions:
#             import random
#             action = random.choice(actions)
#             print(f"步骤 {step+1}: 状态 {current_state} → 动作 {action} (随机策略打破循环)")
#         else:
#             print(f"状态 {current_state} 没有可用动作")
#             break
#     # 如果当前状态有学习到的策略，使用它
#     elif current_state in learned_policy:
#         action = learned_policy[current_state]
#         print(f"步骤 {step+1}: 状态 {current_state} → 动作 {action} (使用学习策略)")
#     else:
#         # 如果没有学习到策略，使用改进的启发式策略
#         actions = gridworld.get_actions(current_state)
#         if not actions:
#             print(f"状态 {current_state} 没有可用动作")
#             break
#
#         # 改进策略：优先选择朝向目标的方向
#         goal = gridworld.goal  # 目标位置 (4, 4)
#         current_x, current_y = current_state
#         goal_x, goal_y = goal
#
#         # 计算到目标的方向
#         dx = goal_x - current_x  # 需要向下移动的距离
#         dy = goal_y - current_y  # 需要向右移动的距离
#
#         # 优先级：朝向目标的方向
#         preferred_actions = []
#         if dx > 0 and 'south' in actions:  # 需要向下
#             preferred_actions.append('south')
#         if dx < 0 and 'north' in actions:  # 需要向上
#             preferred_actions.append('north')
#         if dy > 0 and 'east' in actions:   # 需要向右
#             preferred_actions.append('east')
#         if dy < 0 and 'west' in actions:   # 需要向左
#             preferred_actions.append('west')
#
#         # 如果有朝向目标的动作，选择第一个
#         if preferred_actions:
#             action = preferred_actions[0]
#             print(f"步骤 {step+1}: 状态 {current_state} → 动作 {action} (朝向目标的启发式策略)")
#         else:
#             # 否则选择Q值最高的动作
#             best_action = None
#             best_q = float('-inf')
#             for act in actions:
#                 q_value = qfunction.get_q_value(current_state, act)
#                 if q_value > best_q:
#                     best_q = q_value
#                     best_action = act
#
#             action = best_action
#             print(f"步骤 {step+1}: 状态 {current_state} → 动作 {action} (Q值: {best_q:.4f}, 贪心策略)")
#
#     # 执行动作
#     next_state, reward, done = gridworld.execute(current_state, action)
#     path.append(next_state)
#     current_state = next_state
#
#     if done:
#         print(f"游戏结束！")
#         break
#
# print(f"\n完整路径: {' → '.join(map(str, path))}")
# print(f"路径长度: {len(path)-1} 步")
# print(f"最终状态: {current_state}")
# print(f"是否到达目标: {current_state in gridworld.get_goal_states()}")
