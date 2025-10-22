
from collections import defaultdict

from debugpy.common.timestamp import current
from tenacity import retry

#from main import q_value
from q_learning.q_policy import DeterministicPolicy
class QTable():
    def __init__(self,default_value=0.0):
        self.default_q_value = default_value
        self.q_table = defaultdict(lambda: self.default_q_value)
    #定义Q值表的基本操作
    """ Return the Q-value of action in state """
    def get_q_value(self,  state, action):
        #从表格直接获取Q(s,a)
        return self.q_table.get((state, action), self.default_q_value)
    def set_q_value(self,state, action, q_value):
        self.q_table[(state, action)] = q_value
    def update(self, state, action, target,alpha=0.1):
        """基于蒙特卡洛更新
        根据给定的目标值(target)，使用增量更新规则更新Q值。
        这个函数是TD和MC等多种算法的通用更新核心。

        Q_new = Q_old + alpha * (target - Q_old)

        :param state: 状态
        :param action: 动作
        :param target: 学习的目标值。
                      对于MC更新, target是模拟的最终回报G。
                      对于TD更新, target是 r + gamma * max_a'Q(s',a')。
        :param alpha: 学习率
        """
        current=self.get_q_value(state, action)
        new_q = current + alpha * (target - current)   #q是“总奖励/访问次数”的均值表示，多次更新后是所有target样本回报的运行均值
        self.set_q_value(state, action, new_q)
    def merge(self, other_q_table,weight=0.5):
        """合并其他Q表"""
        assert isinstance(other_q_table,(QTable,dict)), "other_q_table must be QTable or dict"
        if isinstance(other_q_table,QTable):
            other_q_table=other_q_table.q_table
        else:
            other_q_table=other_q_table
        for (state, action),q_value in other_q_table.items():
            #合并
            self.q_table[(state, action)]=self.get_q_value(state, action)*(1-weight)+weight*q_value
    def get_best_q(self,mdp, state):
        """获取某个状态的最大Q值"""
        if state == 'terminal':
            return 0.0
        actions=mdp.get_actions(state)
        if not actions:
            return 0.0
        return max(self.get_q_value(state, action) for action in actions)
    def extract_policy(self, mdp):
        """
        从Q表中提取确定性策略，根据Q表决定策略即当前状态选择什么动作
        return：DeterministicPolicy实例
        """
        policy = DeterministicPolicy()
        for state in mdp.get_states():
            max_q = float("-inf")
            for action in mdp.get_actions(state):
                q_value = self.get_q_value(state, action)

                # If this is the maximum Q-value so far,
                # set the policy for this state
                if q_value > max_q:
                    policy.update(state, action)
                    max_q = q_value

        return policy
    def get_best_action(self,mdp,state):
        """获取当前状态最优动作"""
        if mdp.is_terminal(state):
            return None
        actions=mdp.get_actions(state)
        if not actions:
            return None
        return max(actions,key=lambda a:self.get_q_value(state,a))
    def compute_model_based_q_value(self,mdp,state,action):
        """使用MDP模型动态计算Q值，贝尔曼更新，适合已知环境"""
        q_value=0.0
        transitions=mdp.get_transitions(state,action)   #return (new_state,probability)
        for (new_state,probability) in transitions:
            reward=mdp.get_reward(state,action,new_state)
            next_value=self.get_best_q(mdp,new_state)
            q_value+=probability*(reward+mdp.get_discount_factor()*next_value)
        return q_value
    def batch_updata_from_mdp(self,mdp,num_sweeps):
        """
        批量更新
        :param mdp:environment
        :num_sweeps:scanning times
        """
        for _ in range(num_sweeps):
            new_table={}
            for state in mdp.get_states():
                for action in mdp.get_actions(state):
                    new_table[(state,action)]=self.compute_model_based_q_value(mdp,state,action)

            for (state,action),q_value in new_table.items():
                #self.q_table[(state,action)]=q_value
                self.set_q_value(state,action,q_value)

    def __str__(self):
        result="QTable:\n"
        for (state,action),q_value in self.q_table.items():
            result+="State={},Action={}:q_value={:.4f}\n".format(state,action,q_value)

        return result


if __name__=="__main__":
    from mdp.gridworld import GridWorld
    #create environment
    env=GridWorld(size=3,goal=(2,2))
    #create q-table
    q_table=QTable()
    #visualize q-table and q_function
    env.visualise_policy(q_table.extract_policy(env).policy_table)
    env.visualise_q_function(q_table.q_table)
    q_table.batch_updata_from_mdp(env,1)
    print(q_table)