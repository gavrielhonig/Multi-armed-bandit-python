#####################################################################################
#  This file contains unclean version of all algorithms
#  with additions like:
#  1. MSE - number of mistakes the algorithm made through running
#  2. Cross Entropy - y * log(prob to subscribe) + (1-y) * log(1 - prob to subscribe)
#  Regret - compering Oracle with real performance of algorithm (run the algorithm
#  again with updated parameters in order to see what would happen if we knew
#  the real parameters.)
#####################################################################################
import numpy as np
import pandas as pd
from numpy.linalg import inv
import warnings

warnings.filterwarnings('ignore')
from collections import Counter
import matplotlib.pyplot as plt
import time


# load Lightricks data
arms = np.load('arms.npy')
rewards = np.load('rewards.npy')
contexts = np.load('contexts.npy')
num_of_events = np.load('num_of_events.npy')
num_dims = contexts.shape[1]


class LinUCB:
    def __init__(self, n_arms, n_dims, alpha, true_A=None, true_b=None):
        self.n_arms = n_arms
        self.n_dims = n_dims
        self.alpha = alpha
        self.prob_reward = {}

        self.A = {}  # D.T*D + I for each arm
        self.b = {}  # rewards

        # step 1: param initialization
        # For each arm, create A, b
        for arm in range(1, self.n_arms + 1):
            if arm not in self.A:
                self.A[arm] = np.identity(n_dims)  # A dimensions: d*d
            if arm not in self.b:
                self.b[arm] = np.zeros(n_dims)  # B dimesions: d*1

        if true_A:
            for arm in range(1, self.n_arms + 1):
                if arm not in self.A:
                    self.A[arm] = true_A[arm]
                if arm not in self.b:
                    self.b[arm] = true_b[arm]

    # step 2: compute UCB for each Arm
    def play(self, context, t_round=None):
        UCB = {}

        for arm in range(1, self.n_arms + 1):
            theta_a = np.dot(inv(self.A[arm]), self.b[arm])
            std = np.sqrt(np.linalg.multi_dot([np.transpose(context), np.linalg.inv(self.A[arm]), context]))
            theta_a_context = np.dot(theta_a.T, context)
            self.prob_reward[arm] = 1 / (1 + np.exp(-1 * theta_a_context))
            if self.prob_reward[arm] == 1:
                self.prob_reward[arm] = 0.99

            p_ta = theta_a_context + self.alpha * std

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

    def update_alpha(self, alpha_max):
        self.alpha = alpha_max

    def get_n_arms(self):
        return self.n_arms

    def get_A(self):
        return self.A

    def get_b(self):
        return self.b

    def get_prob_reward(self, arm):
        return self.prob_reward[arm]


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


class GB:
    def __init__(self, n_arms, n_dims, true_beta=None):
        self.n_arms = n_arms  # number of arms
        self.n_dims = n_dims  # number of dimensions

        self.beta = {}
        self.true_beta = {}
        self.s_arm = {}  # clients we already watched
        self.prob_reward = {}

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
            self.prob_reward[arm] = 1 / (1 + np.exp(-1 * p_ta))
            if self.prob_reward[arm] == 1:
                self.prob_reward[arm] = 0.99
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
        # self.s_arm[arm].append(event)
        context = contexts[self.s_arm[arm]]
        x_t_x = np.dot(context.T, context)
        x_t_x = 0.9 * np.cov(x_t_x) + 0.1 * np.mean(np.diag(np.cov(x_t_x))) * np.identity(num_dims)  # regularization
        inverse = np.linalg.inv(x_t_x.reshape(num_dims, num_dims))
        self.beta[arm] = np.dot(np.dot(inverse, context.T), y_odds)  # compute & update beta

    def get_s_arm(self, arm):
        return self.s_arm[arm]

    def get_beta(self):
        return self.beta

    def get_prob_reward(self, arm):
        return self.prob_reward[arm]


def offlineEvaluate(mab, arms, rewards, contexts, num_of_events=num_of_events, n_rounds=None, update=True,
                    algorithm=None):
    h0 = []  # Arms History list
    R0 = []  # Total Reward - If action = play() then we check the reward, either reward=0

    count = 0
    cross_entropy = 0
    mse = 0
    for event in range(num_of_events):

        if len(h0) == n_rounds:  # If reach required number of rounds then stop
            break

        # if len(h0) % 2000 == 0 and len(h0) != 0 and len(h0) <= 10000:  # Update alpha every 2000 rounds till 10k
        #     mab.update_alpha(tune_alpha(len(h0)))

        # Play an arm
        action = mab.play(t_round=len(h0) + 1, context=contexts[event])
        if action == arms[event]:
            if algorithm:
                if np.isnan((1 - rewards[event]) * np.log(1 - mab.get_prob_reward(action))):
                    cross_entropy += rewards[event] * np.log(mab.get_prob_reward(action)) + 0
                else:
                    cross_entropy += rewards[event] * np.log(mab.get_prob_reward(action)) + \
                                     (1 - rewards[event]) * np.log(1 - mab.get_prob_reward(action))
                mse += (np.round(mab.get_prob_reward(action)) - rewards[event]) ** 2
            count += 1
            h0.append(action)
            R0.append(rewards[event])
            if update:
                mab.update(arms[event], rewards[event], contexts[event])
        if algorithm:
            if event == num_of_events - 1:
                cross_entropy = -1 * cross_entropy / len(h0)

    mse = mse / len(h0)

    return R0, h0, cross_entropy, mse


def offlineEvaluate_GB(mab, arms, rewards, contexts, nrounds=None, update=True):
    h0 = []  # Arms History list
    R0 = []  # Total Reward - If action = play() then we check the reward, either reward=0

    count = 0
    cross_entropy = 0
    mse = 0
    for event in range(num_of_events):

        if len(h0) == nrounds:  # If reach required number of rounds then stop
            break

        # Play an arm

        cross_entropy = 0
        mse = 0
        action = mab.play(tround=len(h0) + 1, context_p=contexts[event])
        if action == arms[event]:
            if np.isnan((1 - rewards[event]) * np.log(1 - mab.get_prob_reward(action))):
                cross_entropy += rewards[event] * np.log(mab.get_prob_reward(action)) + 0
            else:
                cross_entropy += rewards[event] * np.log(mab.get_prob_reward(action)) + \
                                 (1 - rewards[event]) * np.log(1 - mab.get_prob_reward(action))
            mse += (np.round(mab.get_prob_reward(action)) - rewards[event]) ** 2
            count += 1
            mab.s_arm[action].append(event)
            if update:
                if action not in h0:
                    mab.update(arms[event], rewards[event], contexts, event)
                else:
                    mab.update(arms[event], rewards[mab.get_s_arm(action)], contexts, event)
            h0.append(action)
            R0.append(rewards[event])

    return R0, h0, cross_entropy, mse


if __name__ == '__main__':
    start_time = time.time()

    mab_linUCB = LinUCB(5, num_dims, 0.1)
    results_LinUCB, arms_chosen_LinUCB, cross_entropy, mse = offlineEvaluate(mab_linUCB, arms, rewards, contexts,
                                                                             num_of_events, algorithm='LinUCB')
    print('LinUCB average reward', "%.4f" % np.mean(results_LinUCB))
    print('CE:', cross_entropy, 'mse:', mse)
    print("--- %.4f seconds ---" % (time.time() - start_time))
    mab_linUCB_t = LinUCB(5, num_dims, 0.1, mab_linUCB.get_A(), mab_linUCB.get_b())
    results_LinUCB_t, arms_chosen_LinUCB_t, cross_entropy2_t, prob_t = offlineEvaluate(mab_linUCB, arms,
                                                                                       rewards, contexts, num_of_events,
                                                                                       update=False, algorithm='LinUCB')

    ##### regret #####
    len_min_LUCB = np.min([len(results_LinUCB), len(results_LinUCB_t)])
    regret_LUCB = np.array(results_LinUCB_t[0:len_min_LUCB]) - np.array(results_LinUCB[0:len_min_LUCB])
    regret_LUCB_total = regret_LUCB.cumsum()
    regret_LUCB = regret_LUCB.cumsum() / (np.arange(len(regret_LUCB)) + 1)

    print('total regret:', np.sum(regret_LUCB))
    plt.plot(np.arange(len_min_LUCB), regret_LUCB, label='LinUCB')
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title('Mean regret per round')
    plt.legend()
    plt.show()

    plt.plot(np.arange(len(regret_LUCB_total)), regret_LUCB_total, label='LinUCB')
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title('Total regret per round')
    plt.legend()
    plt.show()

    c = Counter(arms_chosen_LinUCB)
    print('Arms', c)
