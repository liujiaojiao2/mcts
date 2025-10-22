#Gridworld 实现（继承MDP）
#创建简单网格世界，演示算法
import random
from collections import defaultdict

from matplotlib.patches import Rectangle

#from pyasn1.codec.ber.decoder import stErrorCondition

from mdp.base_mdp import MDP
import numpy as np
import matplotlib.pyplot as plt
class GridWorld(MDP):
    # 动作常量（用于兼容旧逻辑）
    UP = 'north'
    DOWN = 'south'
    LEFT = 'west'
    RIGHT = 'east'
    TERMINATE = 'terminate'
    def __init__(self, size=5, walls=None, start=(0, 0), goal=(4, 4),  action_cost=0, discount_factor=0.9, noise=0.1):
        self.size = size
        self.walls = walls if walls is not None else []
        self.start = start
        self.goal = goal
        self.action_cost = action_cost
        self.discount_factor = discount_factor
        self.noise = noise

        self.TERMINAL = 'terminal'  #stop state
        self.TERMINATE= 'terminate'  #stop action

        self.blocked_states = [] #额外阻挡状态
        self.episode_rewards = []
        self.reset()
    def reset(self):
        # 重置状态
        self.start=self.start
        self.episode_rewards = []
    def get_initial_state(self):
        return self.start
    def get_states(self):
        """
        :return: a list of states and terminal
        """
        valid_states=[(x,y) for x in range(self.size) for y in range(self.size)
                      if (x,y) not in self.walls and (x,y) not in self.blocked_states]
        valid_states.append(self.TERMINAL)
        return valid_states
    def get_actions(self, state):
        if state == self.TERMINAL:
            return []
        x, y = state
        actions = []

        if x > 0 and (x-1, y) not in self.walls:
            actions.append('north')
        if x < self.size-1 and (x+1, y) not in self.walls:
            actions.append('south')
        if y > 0 and (x, y-1) not in self.walls:
            actions.append('west')
        if y < self.size-1 and (x, y+1) not in self.walls:
            actions.append('east')
        return actions
    def execute(self, state, action,path_history):
        """执行一个动作，返回（next_state，reward，done）"""

        #return next_state,reward,done
        # 1. 首先判断当前状态是否已经是终止状态

        if self.is_terminal(state):
            # 如果游戏已经结束，任何动作都不会改变状态，也不会有新奖励。
            return state, 0.0, True

        if state in self.get_goal_states() and action == self.TERMINATE:
            reward=self.get_goal_states().get(state,0.0)
            return self.TERMINAL, reward, True

        # 获取转移概率
        transitions = self.get_transitions(state, action)

        if not transitions:
            # 如果没有定义的转移（例如撞墙），则停在原地，通常给予惩罚或0奖励
            return state, self.get_reward(state, action, state), False

        # 根据概率选择下一个状态
        next_states, probabilities = zip(*transitions)
        next_state = random.choices(next_states, weights=probabilities, k=1)[0]

        reward = self.get_reward(state, action, next_state, path_history)
        # done = next_state == self.TERMINAL
        done = next_state == self.TERMINAL or next_state in self.get_goal_states()

        return next_state, reward, done
    def is_terminal(self, state):
        return state == self.TERMINAL or state in self.get_goal_states()
    def get_discount_factor(self):
        return self.discount_factor   #返回折扣因子
    def get_transitions(self, state, action):
        """
        :param state:
        :param action:
        :return: merge[(state,prob)]
        """
        transitions = [] #(state,概率)

        if state == self.TERMINAL:
            if action == self.TERMINATE:
                return [(self.TERMINAL, 1.0)]
            else:
                return []

        # Probability of not slipping left or right
        straight = 1 - (2 * self.noise)

        (x, y) = state
        if state in self.get_goal_states():
            if action == self.TERMINATE:
                transitions.append((self.TERMINAL, 1.0))

        elif action == self.UP:
            transitions += self.valid_add(state, (x-1, y), straight)
            transitions += self.valid_add(state, (x, y-1), self.noise)  # 向左滑
            transitions += self.valid_add(state, (x, y+1), self.noise)  # 向右滑

        elif action == self.DOWN:
            transitions += self.valid_add(state, (x+1, y), straight)
            transitions += self.valid_add(state, (x, y-1), self.noise)  # 向左滑
            transitions += self.valid_add(state, (x, y+1), self.noise)  # 向右滑

        elif action == self.RIGHT:
            transitions += self.valid_add(state, (x, y+1), straight)
            transitions += self.valid_add(state, (x-1, y), self.noise)  # 向上滑
            transitions += self.valid_add(state, (x+1, y), self.noise)  # 向下滑

        elif action == self.LEFT:
            transitions += self.valid_add(state, (x, y-1), straight)
            transitions += self.valid_add(state, (x-1, y), self.noise)  # 向上滑
            transitions += self.valid_add(state, (x+1, y), self.noise)  # 向下滑

        # Merge any duplicate outcomes
        merged = defaultdict(lambda: 0.0)
        for (state, probability) in transitions:
            merged[state] = merged[state] + probability


        return list(merged.items())
    def visualise_q_function(self,q_function):
        """
           可视化 Q 函数：每个格子显示各动作对应的 Q 值
           :param q_function: dict，形式为 {(x,y,action): q_value}
           """
        q_grid = np.zeros((self.size, self.size))
        action_labels = np.empty((self.size, self.size), dtype=object)

        for x in range(self.size):
            for y in range(self.size):
                state = (x, y)
                if state == self.TERMINAL or state in self.get_goal_states():
                    q_grid[x, y] = float('nan')
                    action_labels[x, y] = ''
                    continue

                actions = self.get_actions(state)
                max_q = -np.inf
                best_action = ''
                for action in actions:
                    q_val = q_function.get((x, y, action), 0)
                    if q_val > max_q:
                        max_q = q_val
                        best_action = action
                q_grid[x, y] = max_q
                action_labels[x, y] = best_action

        fig, ax = plt.subplots()
        cax = ax.imshow(q_grid, cmap='viridis', interpolation='none')

        # 绘制墙壁
        for wall in self.walls:
            x, y = wall
            ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, facecolor='gray'))

        # 绘制起点和终点
        ax.text(self.start[1], self.start[0], 'S', va='center', ha='center', fontsize=12, color='white')
        ax.text(self.goal[1], self.goal[0], 'G', va='center', ha='center', fontsize=12, color='white')

        # 绘制Q值和动作标签
        for i in range(self.size):
            for j in range(self.size):
                val = q_grid[i, j]
                if np.isnan(val):
                    continue
                label = action_labels[i, j]
                ax.text(j, i, f'{label}\n{val:.2f}', va='center', ha='center', fontsize=8, color='black')

        # 设置坐标轴
        ax.set_xticks(np.arange(-0.5, self.size, 1))
        ax.set_yticks(np.arange(-0.5, self.size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        plt.colorbar(cax)
        plt.title("Q-Function Visualization")
        plt.show()
    def visualise_policy(self,policy):
        """
           可视化策略：每个格子显示动作方向的箭头
           :param policy: dict，形式为 {(x,y): action}
           """
        grid = np.zeros((self.size, self.size))
        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='Greys', interpolation='none')

        # 绘制墙壁
        for wall in self.walls:
            x, y = wall
            ax.add_patch(Rectangle((y - 0.5, x - 0.5), 1, 1, facecolor='gray'))

        # 绘制起点和终点
        ax.text(self.start[1], self.start[0], 'S', va='center', ha='center', fontsize=12)
        ax.text(self.goal[1], self.goal[0], 'G', va='center', ha='center', fontsize=12)

        # 绘制策略箭头
        for state in policy:
            if self.is_terminal(state):
                continue
            x, y = state
            action = policy[state]
            dx, dy = 0, 0
            # if action == 'north':
            #     dx, dy = 0, -0.4
            # elif action == 'south':
            #     dx, dy = 0, 0.4
            # elif action == 'west':
            #     dx, dy = -0.4, 0
            # elif action == 'east':
            #     dx, dy = 0.4, 0
            if action == 'north':  # 北 → 上（x-）
                dx, dy = -0.4, 0
            elif action == 'south':  # 南 → 下（x+）
                dx, dy = 0.4, 0
            elif action == 'west':  # 西 → 左（y-）
                dx, dy = 0, -0.4
            elif action == 'east':  # 东 → 右（y+）
                dx, dy = 0, 0.4
            else:
                continue  # 忽略 TERMINATE 等无效动作

            ax.arrow(y, x, dy, dx, head_width=0.2, length_includes_head=True, color='blue')

        # 设置坐标轴
        ax.set_xticks(np.arange(-0.5, self.size, 1))
        ax.set_yticks(np.arange(-0.5, self.size, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        plt.title("Policy Visualization")
        plt.show()
    def valid_add(self, state, new_state, probability):
        # If the next state is blocked, stay in the same state
        if probability == 0.0:
            return []

        if new_state in self.blocked_states:
            return [(state, probability)]

        # Move to the next space if it is not off the grid
        (x, y) = new_state
        # if x >= 0 and x < self.width and y >= 0 and y < self.height:
        #     return [((x, y), probability)]
        if 0 <= x < self.size and 0 <= y < self.size:
            return [(new_state, probability)]
        else:
            return [(state, probability)]
    def get_reward(self, state, action, new_state,path_history):
        # 定义在状态state时，执行action后，立刻返回的单步信号，即使评价
        reward=0.0
        if new_state in self.get_goal_states().keys():
            reward=self.get_goal_states().get(new_state)
        else:
            if new_state in path_history:
                reward=-5.0
            else:

                # 2. 计算距离变化带来的启发式奖励
                current_dist = self.manhattan_distance(state, self.goal)
                new_dist = self.manhattan_distance(new_state, self.goal)

                # 每靠近一步，就给予一个小的正奖励 (例如 +0.5)
                # 每远离一步，就给予一个小的负奖励 (例如 -2)
                # 平行移动时，也给予负奖励，以免原地打转
                distance_reward = current_dist - new_dist
                if distance_reward > 0:
                    distance_reward=distance_reward*0.5
                else:
                    distance_reward=0

                # 3. 结合固定的行动成本
                reward = distance_reward + self.action_cost  # 例如: 0.1 + (-0.01) = +0.09

        step=len(self.episode_rewards)
        self.episode_rewards+=[reward*(self.discount_factor ** step)]
        return reward

    def manhattan_distance(self, pos1, pos2):
        """计算两个坐标点之间的曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    def get_goal_states(self):
        # Returns a dictionary of goal states and their rewards
        return {self.goal: 100.0}

# 示例：测试 GridWorld 基本功能
# if __name__ == '__main__':
#
#     env = GridWorld(size=5, goal=(3, 3))
#     print("初始状态:", env.get_initial_state())
#     print("目标状态奖励:", env.get_goal_states())
#     print("折扣因子:", env.get_discount_factor())
#
#     state = env.get_initial_state()
#     action = 'south'
#     next_state, reward, done = env.execute(state, action)
#     print(f"执行动作 {action} 后: 下一状态={next_state}, 奖励={reward}, 是否结束={done}")
#
#     print("是否终止状态:", env.is_terminal(next_state))
# #    示例策略
#     example_policy = {
#         (0, 0): 'south',
#         (0, 1): 'east',
#         (1, 0): 'south',
#         (1, 1): 'east',
#         (3, 3): 'terminate'
#     }
#
#     # 示例 Q 函数
#     example_q_function = {
#         (0, 0, 'south'): 0.5,
#         (0, 0, 'east'): 0.3,
#         (0, 1, 'east'): 0.7,
#         (0, 1, 'south'): 0.6,
#         (1, 0, 'south'): 0.9,
#         (1, 0, 'east'): 0.2,
#         (1, 1, 'east'): 0.8,
#         (3, 3, 'terminate'): 1.0
#     }
#
#     env.visualise_policy(example_policy)
#     env.visualise_q_function(example_q_function)