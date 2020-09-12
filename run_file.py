#################################################
# All algorithms: Greedy bandit, Epsilon Greedy,
# UCB, & LinUCB
#################################################

from numpy.linalg import inv
import warnings
import numpy as np

warnings.filterwarnings('ignore')
from collections import Counter
import matplotlib.pyplot as plt
import time


# arms = np.load('arms.npy')
# rewards = np.load('rewards.npy')
# contexts = np.load('contexts.npy')
# num_of_events = np.load('num_of_events.npy')


class LinUCB():
    def __init__(self, n_arms, n_dims, alpha):
        self.n_arms = n_arms
        self.n_dims = n_dims
        self.alpha = alpha

        self.A = {}  # D.T*D + I for each arm
        self.b = {}  # rewards

        # step 1: param initialization
        # For each arm, create A, b
        for arm in range(1, self.n_arms + 1):
            if arm not in self.A:
                self.A[arm] = np.identity(n_dims)  # A dimensions: d*d
            if arm not in self.b:
                self.b[arm] = np.zeros(n_dims)  # B dimesions: d*1

    # step 2: compute UCB for each Arm
    def play(self, context, t_round=None):
        UCB = {}

        for arm in range(1, self.n_arms + 1):
            theta_a = np.dot(inv(self.A[arm]), self.b[arm])
            std = np.sqrt(np.linalg.multi_dot([np.transpose(context), np.linalg.inv(self.A[arm]), context]))
            p_ta = np.dot(theta_a.T, context) + self.alpha * std
            if not np.isnan(p_ta):  # make sure the result of calculation is valid number
                UCB[arm] = p_ta

        # step 3: take action
        max_UCB = max(UCB.values())
        max_UCB_key = [key for key, value in UCB.items() if value == max_UCB]
        if len(max_UCB_key) > 1:
            action = np.random.choice(max_UCB_key)  # Tie Breaker
        else:
            action = max_UCB_key[0]
        return action

    # step 4: update
    def update(self, arm, reward, context):
        self.A[arm] = np.add(self.A[arm], np.outer(context, np.transpose(context)))
        self.b[arm] = np.add(self.b[arm], np.dot(reward, context))


class UCB():
    def __init__(self, n_arms, rho, Q_0=np.inf):
        self.n_arms = n_arms
        self.rho = rho
        self.Q_0 = Q_0
        self.arm_visit_count = {}
        self.arm_total_reward = {}

        self.arm_with_avg_reward = {}
        for arm in range(1, self.n_arms + 1):
            self.arm_with_avg_reward[arm] = self.Q_0  # Initial all the arm with Q0

            self.arm_visit_count[arm] = 0  # Initial all the arm with zero number of visits
            self.arm_total_reward[arm] = 0  # Initial all the arm with zero reward

    def play(self, t_round, context=None):
        temp_arm_with_Q = self.arm_with_avg_reward

        for arm in temp_arm_with_Q:
            if self.arm_visit_count[arm] == 0:  # Use Q0 for the first round
                continue


            else:
                # At t_round, calculate Q with exlpore boost for each arm
                explore_boost_const = self.rho * np.log(t_round) / self.arm_visit_count[arm]

                temp_arm_with_Q[arm] = temp_arm_with_Q[arm] + np.sqrt(explore_boost_const)

        # Getting the highest value from Q, then find the corresponding key and append them
        highest = max(temp_arm_with_Q.values())
        highest_Qs = [key for key, value in temp_arm_with_Q.items() if value == highest]
        if len(highest_Qs) > 1:
            action = np.random.choice(highest_Qs)  # Tie Breaker
        else:
            action = highest_Qs[0]
        return action

    def update(self, arm, reward, context=True):
        self.arm_visit_count[arm] += 1
        self.arm_total_reward[arm] += reward
        updated_reward = self.arm_total_reward[arm] / self.arm_visit_count[arm]

        self.arm_with_avg_reward.update({arm: updated_reward})

        return self.arm_with_avg_reward


class GB():
    def __init__(self, n_arms, n_dims, true_beta=None):
        self.n_arms = n_arms  # number of arms
        self.n_dims = n_dims  # number of dimensions

        self.beta = {}
        self.true_beta = {}
        self.s_arm = {}  # clients we already watched

        # step 1: param initialization
        # For each arm, create beta, s_arm
        for arm in range(1, self.n_arms + 1):
            if arm not in self.s_arm:
                self.s_arm[arm] = []
            if not self.true_beta:
                if arm not in self.beta:
                    self.beta[arm] = np.zeros(n_dims)
            else:
                if arm not in self.beta:
                    self.beta[arm] = true_beta[arm]

    # step 2: compute expectation for each Arm
    def play(self, context_p, tround=None):
        p_t = {}

        for arm in range(1, self.n_arms + 1):
            p_ta = np.dot(context_p, self.beta[arm])
            if not np.isnan(p_ta):  # make sure the result of calculation is valid number
                p_t[arm] = p_ta

        # step 3: take action
        max_GB = max(p_t.values())
        max_GB_key = [key for key, value in p_t.items() if value == max_GB]
        if len(max_GB_key) > 1:
            action = np.random.choice(max_GB_key)  # Tie Breaker
        else:
            action = max_GB_key[0]

        return action

    # step 4: update
    def update(self, arm, reward, contexts, event):
        y_sigmoid = 1 / (1 + np.exp(-reward))  # logistic regression
        y_odds = np.log(y_sigmoid / (1 - y_sigmoid))
        context = contexts[self.s_arm[arm]]
        x_t_x = np.dot(context.T, context)
        x_t_x = 0.8 * np.cov(x_t_x) + 0.2 * np.mean(np.diag(np.cov(x_t_x))) * np.identity(len(x_t_x))  # regularization
        inverse = np.linalg.inv(x_t_x.reshape(len(x_t_x), len(x_t_x)))
        self.beta[arm] = np.dot(np.dot(inverse, context.T), y_odds)  # compute & update beta

    def get_s_arm(self, arm):
        return self.s_arm[arm]

    def get_beta(self):
        return self.beta


class GreedyEpsilon():
    def __init__(self, n_arms, epsilon, Q0=np.inf):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.Q0 = Q0
        self.arm_visit_count = {}
        self.arm_total_reward = {}

        self.arm_with_Q = {}
        for arm in range(1, self.n_arms + 1):
            self.arm_with_Q[arm] = self.Q0  # Initial all the arm with Q0

            self.arm_visit_count[arm] = 0  # Initial all the arm with zero number of visits
            self.arm_total_reward[arm] = 0  # Initial all the arm with zero reward

    def play(self, context=None, t_round=None):
        if np.random.random() < self.epsilon:  # exploration(Random select an arm)
            action = np.random.choice(range(1, self.n_arms + 1))
        else:
            highest = max(self.arm_with_Q.values())
            highest_Qs = [key for key, value in self.arm_with_Q.items() if value == highest]
            if len(highest_Qs) > 1:
                action = np.random.choice(highest_Qs)  # Tie Breaker
            else:
                action = highest_Qs[0]

        return action

    def update(self, arm, reward, context=None):
        self.arm_visit_count[arm] += 1
        self.arm_total_reward[arm] += reward
        updated_reward = self.arm_total_reward[arm] / self.arm_visit_count[arm]
        self.arm_with_Q.update({arm: updated_reward})
        return self.arm_with_Q


def offlineEvaluate(mab, arms, rewards, contexts, num_of_events, n_rounds=None):
    h0 = []  # Arms History list
    R0 = []  # Total Reward - If action = play() then we check the reward, and update the model.

    count = 0
    for event in range(num_of_events):

        if event == n_rounds:  # If reach required number of rounds then stop
            break

        # Play an arm
        action = mab.play(t_round=len(h0) + 1, context=contexts[event])
        if action == arms[event]:
            count += 1
            h0.append(action)
            R0.append(rewards[event])
            mab.update(arms[event], rewards[event], contexts[event])

    return R0, h0


def offlineEvaluate_GB(mab, arms, rewards, contexts, num_of_events, nrounds=None):
    h0 = []  # Arms History list
    R0 = []  # Total Reward - If action = play() then we check the reward, either reward=0

    count = 0
    for event in range(num_of_events):

        if len(h0) == nrounds:  # If reach required number of rounds then stop
            break

        # Play an arm
        action = mab.play(tround=len(h0) + 1, context_p=contexts[event])
        if action == arms[event]:
            count += 1
            mab.s_arm[action].append(event)
            if action not in h0:
                mab.update(arms[event], rewards[event], contexts, event)
            else:  # Update based on all data we have so far
                mab.update(arms[event], rewards[mab.get_s_arm(action)], contexts, event)
            h0.append(action)
            R0.append(rewards[event])

    return R0, h0


def onlineEvaluate(mab, contexts, p1, p2, num_of_events, n_rounds=None):
    h0 = []  # Arms History list
    R0 = []  # Total Reward - If action = play() then we check the reward, and update the model.

    count = 0
    for event in range(num_of_events):

        if event == n_rounds:  # If reach required number of rounds then stop
            break

        # Play an arm
        action = mab.play(t_round=len(h0) + 1, context=contexts[event])
        if action == 1 or action == 2 or action == 3:
            rewards = np.random.binomial(1, p1, 1)
        else:
            rewards = np.random.binomial(1, p2, 1)
        count += 1
        h0.append(action)
        R0.append(rewards)
        mab.update(action, int(rewards), contexts[event])

    return R0, h0


if __name__ == '__main__':
    arms = np.load('arms.npy')
    rewards = np.load('rewards.npy')
    contexts = np.load('contexts.npy')
    num_of_events = np.load('num_of_events.npy')

    start_time = time.time()

    """Offline"""

    mab_linUCB = LinUCB(5, contexts.shape[1], 0.1)
    results_LinUCB, arms_chosen_LinUCB = offlineEvaluate(mab_linUCB, arms, rewards, contexts, num_of_events)
    print('LinUCB average reward', "%.4f" % np.mean(results_LinUCB))
    print('LinUCB update', len(results_LinUCB), 'times')
    print("--- %.4f seconds ---" % (time.time() - start_time))
    c = Counter(arms_chosen_LinUCB)
    print('Arms', c)

    """Online"""
    # mab_linUCB = LinUCB(5, 4, 0.1)
    # results_LinUCB, arms_chosen_LinUCB = onlineEvaluate(mab_linUCB, contexts, 0.3,0.6,num_of_events)
    # print('LinUCB average reward', "%.4f" % np.mean(results_LinUCB))
    # print('LinUCB update', len(results_LinUCB), 'times')
    # c = Counter(arms_chosen_LinUCB)
    # print('Arms',c)
