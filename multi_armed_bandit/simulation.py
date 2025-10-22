"""
测试对比不同多臂老虎机算法，主要为ucb
"""
from collections import defaultdict
import random
from q_learning.q_table import QTable
from ucb import UpperConfidenceBounds

""" Run a bandit algorithm for a number of episodes, with each episode
being a set length.
"""

def run_bandit(bandit, episodes=200, episode_length=500, drift=True):

    # The actions available
    actions = [0, 1, 2, 3, 4]

    # A dummy state
    state = 1

    rewards = []
    for _ in range(0, episodes):
        bandit.reset()

        # The probability of receiving a payoff of 1 for each action
        probabilities = [0.1, 0.3, 0.7, 0.2, 0.1]

        # The number of times each arm has been selected
        times_selected = defaultdict(lambda: 0)
        qtable = QTable()

        episode_rewards = []
        for step in range(episode_length):

            # Halfway through the episode, change the probabilities
            if drift and step == episode_length / 2:
                probabilities = [0.5, 0.2, 0.0, 0.3, 0.3]

            # Select an action using the bandit
            action = bandit.select(state, actions, qtable)

            # Get the reward for that action
            reward = 0
            if random.random() < probabilities[action]:
                reward = 5

            episode_rewards += [reward]

            times_selected[action] = times_selected[action] + 1
            # qtable.update(
            #     state,
            #     action,
            #     (reward / times_selected[action])
            #     - (qtable.get_q_value(state, action) / times_selected[action]),
            # )
            n=times_selected[action]
            qtable.update(
                state,
                action,
                reward,
                1/n
            )


        rewards += [episode_rewards]

    return rewards

def plot_rewards(rewards):
    """
    Plot the mean rewards per step
    :param rewards:List of episode rewards
    """
    import numpy as np
    import matplotlib.pyplot as plt
    mean_rewards_per_step = np.mean(rewards, axis=0)
    plt.plot(mean_rewards_per_step,  label="UCB",color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Mean reward per step")
    plt.title("UCB performance over steps")
    plt.legend()
    plt.grid(True)
    plt.show()

if  __name__ == "__main__":
    bandit = UpperConfidenceBounds()
    rewards = run_bandit(bandit)
    plot_rewards(rewards)