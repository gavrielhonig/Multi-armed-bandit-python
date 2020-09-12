#########################################################
# Simulation on data sampled from normal distribution.
# The idea is to examine convergence by type of variables
# (dummies and continuous variables), and number of arms.
#########################################################
import numpy as np
import pandas as pd
from run_file_additions import LinUCB, offlineEvaluate
from collections import Counter
import matplotlib.pyplot as plt

n = 40000
n_arms = 3
num_of_events = n_arms * n

arms1 = np.full(n, 1)
arms2 = np.full(n, 2)
arms3 = np.full(n, 3)
arms4 = np.full(n, 4)
arms5 = np.full(n, 5)
arms_2 = np.concatenate([arms1, arms2])
arms = np.concatenate([arms1, arms2, arms3])
arms_4 = np.concatenate([arms1, arms2, arms3, arms4])
arms_5 = np.concatenate([arms1, arms2, arms3, arms4, arms5])

res_reward2 = []
res_mean2 = []
res = []
res_reward = []
res_mean = []
res_reward4 = []
res_mean4 = []
res_reward5 = []
res_mean5 = []
res_reward_dum = []
res_mean_dum = []
res_reward_all_dum = []
res_mean_all_dum = []
MSE = []
CE = []
MSE_all_dum = []
CE_all_dum = []
MSE_5 = []
CE_5 = []
num_take_count = 20000  # take the same number of observations from each simulation
num_iter = 8  # number of iterations
for i in range(num_iter):
    y1 = np.random.binomial(1, 0.6, int(n))  # set reward vector for each arm
    y2 = np.random.binomial(1, 0.4, int(n))
    y3 = np.random.binomial(1, 0.2, int(n))
    y4 = np.random.binomial(1, 0.15, int(n))
    y5 = np.random.binomial(1, 0.1, int(n))
    rewards2 = np.concatenate([y2, y3])
    rewards = np.concatenate([y1, y2, y3])
    rewards4 = np.concatenate([y1, y2, y3, y4])
    rewards5 = np.concatenate([y1, y2, y3, y4, y5])

    x1 = np.random.normal(6, 1, int(n))  # set context vector for each arm
    x2 = np.random.normal(4, 1, int(n))
    x3 = np.random.normal(2, 1, int(n))
    x4 = np.random.normal(1, 1, int(n))
    x5 = np.random.normal(0.5, 1, int(n))

    x_a2 = np.concatenate([x1, x2])
    x_a = np.concatenate([x1, x2, x3])
    x_a4 = np.concatenate([x1, x2, x3, x4])
    x_a5 = np.concatenate([x1, x2, x3, x4, x5])

    x1 = np.random.normal(6, 1.2, int(n))  # set context vector for each arm
    x2 = np.random.normal(4, 1.2, int(n))
    x3 = np.random.normal(2, 1.2, int(n))
    x4 = np.random.normal(1, 1.2, int(n))
    x5 = np.random.normal(0.5, 1.2, int(n))

    x_b2 = np.concatenate([x1, x2])
    x_b = np.concatenate([x1, x2, x3])
    x_b4 = np.concatenate([x1, x2, x3, x4])
    x_b5 = np.concatenate([x1, x2, x3, x4, x5])

    x1 = np.random.normal(6, 1.5, int(n))  # set context vector for each arm
    x2 = np.random.normal(4, 1.5, int(n))
    x3 = np.random.normal(2, 1.5, int(n))
    x4 = np.random.normal(1, 1.5, int(n))
    x5 = np.random.normal(0.5, 1.5, int(n))

    # dummies
    x1dum = np.random.binomial(1, 0.6, int(n))  # set context vector for each arm
    x2dum = np.random.binomial(1, 0.4, int(n))
    x3dum = np.random.binomial(1, 0.2, int(n))
    x4dum = np.random.binomial(1, 0.1, int(n))
    x5dum = np.random.binomial(1, 0.05, int(n))

    x1dum2 = np.random.binomial(1, 0.7, int(n))  # set context vector for each arm
    x2dum2 = np.random.binomial(1, 0.5, int(n))
    x3dum2 = np.random.binomial(1, 0.3, int(n))
    x4dum2 = np.random.binomial(1, 0.1, int(n))
    x5dum2 = np.random.binomial(1, 0.05, int(n))

    x1dum3 = np.random.binomial(1, 0.6, int(n))  # set context vector for each arm
    x2dum3 = np.random.binomial(1, 0.45, int(n))
    x3dum3 = np.random.binomial(1, 0.18, int(n))
    x4dum3 = np.random.binomial(1, 0.15, int(n))
    x5dum3 = np.random.binomial(1, 0.08, int(n))

    x1dum4 = np.random.binomial(1, 0.65, int(n))  # set context vector for each arm
    x2dum4 = np.random.binomial(1, 0.42, int(n))
    x3dum4 = np.random.binomial(1, 0.24, int(n))
    x4dum4 = np.random.binomial(1, 0.13, int(n))
    x5dum4 = np.random.binomial(1, 0.09, int(n))

    x_c2 = np.concatenate([x1, x2])
    x_c = np.concatenate([x1, x2, x3])
    x_c4 = np.concatenate([x1, x2, x3, x4])
    x_c5 = np.concatenate([x1, x2, x3, x4, x5])
    x_dum = np.concatenate([x1dum, x2dum, x3dum, x4dum, x5dum])
    x_dum_b = np.concatenate([x1dum2, x2dum2, x3dum2, x4dum2, x5dum2])
    x_dum_c = np.concatenate([x1dum3, x2dum3, x3dum3, x4dum3, x5dum3])
    x_dum_d = np.concatenate([x1dum4, x2dum4, x3dum4, x4dum4, x5dum4])

    x1 = np.random.binomial(1, 0.5, int(n))  # set context vector for each arm
    x2 = np.random.binomial(1, 0.25, int(n))
    x3 = np.random.binomial(1, 0.15, int(n))
    x4 = np.random.binomial(1, 0.1, int(n))
    x5 = np.random.binomial(1, 0.08, int(n))

    x_d2 = np.concatenate([x1, x2])
    x_d = np.concatenate([x1, x2, x3])
    x_d4 = np.concatenate([x1, x2, x3, x4])
    x_d5 = np.concatenate([x1, x2, x3, x4, x5])

    # 2 arms
    df = pd.DataFrame()  # creare data frame containg all data
    df['x_a'] = x_a
    df['x_b'] = x_b
    df['x_c'] = x_c
    df['x_d'] = x_d
    df['arms'] = arms
    df['rewards'] = rewards
    df = df.sample(frac=1)  # sample data to enable the algorithm to expose to every arm
    df = np.array(df)
    contexts, arms, rewards = df[:, 0:int(df.shape[1] - 2)], df[:, int(df.shape[1] - 2)], df[:, int(df.shape[1] - 1)]

    # 3 arms
    df2 = pd.DataFrame()
    df2['x_a'] = x_a2
    df2['x_b'] = x_b2
    df2['x_c'] = x_c2
    df2['x_d'] = x_d2
    df2['arms'] = arms_2
    df2['rewards'] = rewards2
    df2 = df2.sample(frac=1)
    df2 = np.array(df2)
    contexts_2, arms__2, rewards_2 = df2[:, 0:int(df.shape[1] - 2)], df2[:, int(df.shape[1] - 2)], df2[:,
                                                                                                   int(df.shape[1] - 1)]
    # 4 arms
    df4 = pd.DataFrame()
    df4['x_a'] = x_a4
    df4['x_b'] = x_b4
    df4['x_c'] = x_c4
    df4['x_d'] = x_d4
    df4['arms'] = arms_4
    df4['rewards'] = rewards4
    df4 = df4.sample(frac=1)
    df4 = np.array(df4)
    contexts_4, arms__4, rewards_4 = df4[:, 0:int(df.shape[1] - 2)], df4[:, int(df.shape[1] - 2)], df4[:,
                                                                                                   int(df.shape[1] - 1)]
    # 5 arms - 4 variables: all continuous
    df5 = pd.DataFrame()
    df5['x_a'] = x_a5
    df5['x_b'] = x_b5
    df5['x_c'] = x_c5
    df5['x_d'] = x_d5
    df5['arms'] = arms_5
    df5['rewards'] = rewards5
    df5 = df5.sample(frac=1)
    df5 = np.array(df5)
    contexts_5, arms__5, rewards_5 = df5[:, 0:int(df.shape[1] - 2)], df5[:, int(df.shape[1] - 2)], df5[:,
                                                                                                   int(df.shape[1] - 1)]
    # 5 arms - 4 variables: 1 dummy and 3 continuous variables
    df_dum = pd.DataFrame()
    df_dum['x_a'] = x_a5
    df_dum['x_b'] = x_b5
    df_dum['x_c'] = x_dum
    df_dum['x_d'] = x_d5
    df_dum['arms'] = arms_5
    df_dum['rewards'] = rewards5
    df_dum = df_dum.sample(frac=1)
    df_dum = np.array(df_dum)
    contexts_dum, arms__dum, rewards_dum = df_dum[:, 0:int(df.shape[1] - 2)], df_dum[:, int(df.shape[1] - 2)], df_dum[:,
                                                                                                               int(
                                                                                                                   df.shape[
                                                                                                                       1] - 1)]
    # 5 arms - 4 variables: all_dummies
    df_all_dum = pd.DataFrame()
    df_all_dum['x_a'] = x_dum
    df_all_dum['x_b'] = x_dum_b
    df_all_dum['x_c'] = x_dum_c
    df_all_dum['x_d'] = x_dum_d
    df_all_dum['arms'] = arms_5
    df_all_dum['rewards'] = rewards5
    df_all_dum = df_all_dum.sample(frac=1)
    df_all_dum = np.array(df_all_dum)
    contexts_all_dum, arms__all_dum, rewards_all_dum = df_all_dum[:, 0:int(df.shape[1] - 2)], df_all_dum[:, int(
        df.shape[1] - 2)], df_all_dum[:,
                           int(
                               df.shape[
                                   1] - 1)]

    # run MAB on each data frame

    mab2 = LinUCB(2, df.shape[1] - 2, 0.02)
    results_LinUCB2, arms_chosen_LinUCB2, cross_entropy2, mse2 = offlineEvaluate(mab2, arms__2, rewards_2, contexts_2,
                                                                                 n * 2)

    mab = LinUCB(3, df.shape[1] - 2, 0.02)
    results_LinUCB, arms_chosen_LinUCB, cross_entropy, mse = offlineEvaluate(mab, arms, rewards, contexts,
                                                                             num_of_events)

    mab4 = LinUCB(4, df.shape[1] - 2, 0.02)
    results_LinUCB4, arms_chosen_LinUCB4, cross_entropy4, mse4 = offlineEvaluate(mab4, arms__4, rewards_4, contexts_4,
                                                                                 n * 4)

    mab5 = LinUCB(5, df.shape[1] - 2, 0.02)
    results_LinUCB5, arms_chosen_LinUCB5, cross_entropy5, mse5 = offlineEvaluate(mab5, arms__5, rewards_5, contexts_5,
                                                                                 n * 5)

    mab_dum = LinUCB(5, df.shape[1] - 2, 0.02)
    results_LinUCB_dum, arms_chosen_LinUCB_dum, cross_entropy_dum, mse_dum = offlineEvaluate(mab_dum, arms__dum,
                                                                                             rewards_dum,
                                                                                             contexts_dum, n * 5)

    mab_all_dum = LinUCB(5, df.shape[1] - 2, 0.02)
    results_LinUCB_all_dum, arms_chosen_LinUCB_all_dum, cross_entropy_all_dum, mse_all_dum = offlineEvaluate(mab_dum,
                                                                                                             arms__all_dum,
                                                                                                             rewards_all_dum,
                                                                                                             contexts_all_dum,
                                                                                                             n * 5)

    print(Counter(arms_chosen_LinUCB_dum))
    res.append(np.mean(results_LinUCB5))
    res_reward2.append(results_LinUCB2[0:num_take_count])
    res_reward.append(results_LinUCB[0:num_take_count])
    res_reward4.append(results_LinUCB4[0:num_take_count])
    res_reward5.append(results_LinUCB5[0:num_take_count])
    res_reward_dum.append(results_LinUCB_dum[0:num_take_count])
    res_reward_all_dum.append(results_LinUCB_all_dum[0:num_take_count])
    MSE.append(cross_entropy_dum)
    CE.append(mse_dum)
    MSE_all_dum.append(cross_entropy_all_dum)
    CE_all_dum.append(mse_all_dum)
    MSE_5.append(cross_entropy5)
    CE_5.append(mse5)
    if len(res) % num_iter == 0:
        #  compute the average of all iterations at every round
        res_mean2.append(np.array(res_reward2).mean(axis=0))
        res_mean.append(np.array(res_reward).mean(axis=0))
        res_mean4.append(np.array(res_reward4).mean(axis=0))
        res_mean5.append(np.array(res_reward5).mean(axis=0))
        res_mean_dum.append(np.array(res_reward_dum).mean(axis=0))
        res_mean_all_dum.append(np.array(res_reward_all_dum).mean(axis=0))
        res_reward2 = []
        res_reward = []
        res_reward4 = []
        res_reward5 = []
        res_reward_dum = []
        res_reward_all_dum = []


cum_reward2 = []
for i in range(num_take_count):
    cum_reward2.append(np.mean(res_mean2[0][0:i + 1]))

cum_reward = []
for i in range(num_take_count):
    cum_reward.append(np.mean(res_mean[0][0:i + 1]))

cum_reward4 = []
for i in range(num_take_count):
    cum_reward4.append(np.mean(res_mean4[0][0:i + 1]))

cum_reward5 = []
for i in range(num_take_count):
    cum_reward5.append(np.mean(res_mean5[0][0:i + 1]))

cum_reward_dum = []
for i in range(num_take_count):
    cum_reward_dum.append(np.mean(res_mean_dum[0][0:i + 1]))

cum_reward_all_dum = []
for i in range(num_take_count):
    cum_reward_all_dum.append(np.mean(res_mean_all_dum[0][0:i + 1]))

#  Type of variables
plt.plot(np.arange(len(cum_reward5)), cum_reward5, label='Continuous variables')
plt.plot(np.arange(len(cum_reward_dum)), cum_reward_dum, label='Continuous & dummy variables')
plt.plot(np.arange(len(cum_reward_all_dum)), cum_reward_all_dum, label='Dummy variables')
plt.xlabel('Round')
plt.ylabel('Mean reward')
plt.title("Mean reward by round and arms' variables type")
plt.legend()
plt.show()

# # MSE & CE
# plt.plot(MSE, res, label='MSE')
# plt.plot(CE, res, label='cross entropy')
# plt.xlabel('Mean reward')
# plt.ylabel('Total mean reward')
# plt.title("Total mean reward by cross entropy & MSE")
# plt.legend()
# plt.show()

# Mean reward by round and number of arms
plt.plot(np.arange(len(cum_reward2)), cum_reward2, label='2 arms')
plt.plot(np.arange(len(cum_reward)), cum_reward, label='3 arms')
plt.plot(np.arange(len(cum_reward4)), cum_reward4, label='4 arms')
plt.plot(np.arange(len(cum_reward5)), cum_reward5, label='5 arms')
plt.xlabel('Round')
plt.ylabel('Mean reward')
plt.title("Mean reward by round and number of arms")
plt.legend()
plt.show()


#########################################################
# Simulation on data sampled from multivariate
# normal distribution. The idea is to sample reward
# of the first arm from bernoulli distribution and then
# to replace some of the rewards from zero to 1.
# again by sampling from bernoulli distribution.
#########################################################
def sample_rewards(array, rate=0.25):
    """
    :param array:vector of rewards
    :param rate: probability of bernoulli distribution
    :return: vector of rewards with higher probability to subscribe
    """
    zeros = []
    for i in range(len(array)):
        if array[i] == 0:
            zeros.append(i)
    array[zeros] = np.random.binomial(1, rate, int(len(zeros)))
    return array


n = 40000
n_arms = 8
num_of_events = n_arms * n

arms1 = np.full(n, 1)
arms2 = np.full(n, 2)
arms3 = np.full(n, 3)
arms = np.concatenate([arms1, arms2, arms3])

num_take_count = 20000  # take the same number of observations from each simulation
num_iter = 3  # number of iterations

res = []
res_reward = []
res_mean = []
for i in range(num_iter):
    y1 = np.random.binomial(1, 0.2, int(n))  # set reward vector for each arm
    y2 = sample_rewards(y1)
    y3 = sample_rewards(y2, rate=0.333)
    reward = np.concatenate([y1, y2, y3])

    mean = [0, 1, 4, 7]
    cov = np.identity(4)
    context = np.random.multivariate_normal(mean, cov, n)
    context = np.concatenate([context, context, context])
    df = pd.DataFrame(context)
    df['arms'] = arms
    df['reward'] = reward
    df = df.sample(frac=1)  # sample data to enable the algorithm to expose to every arm
    df = np.array(df)
    contexts, arms, rewards = df[:, 0:int(df.shape[1] - 2)], df[:, int(df.shape[1] - 2)], df[:, int(df.shape[1] - 1)]

    mab = LinUCB(n_arms, df.shape[1] - 2, 0.02)
    results_LinUCB, arms_chosen_LinUCB, cross_entropy, mse = offlineEvaluate(mab, arms, rewards, contexts,
                                                                             num_of_events)

    mab_ucb = LinUCB(n_arms, df.shape[1] - 2, 0.02)
    results_UCB, arms_chosen_UCB, cross_entropy_ucb, mse_ucb = offlineEvaluate(mab_ucb, arms, rewards, contexts,
                                                                             num_of_events)

    res.append(np.mean(results_LinUCB))
    res_reward.append(results_LinUCB[0:num_take_count])
    if len(res) % num_iter == 0:
        #  compute the average of all iterations at every round
        res_mean.append(np.array(res_reward).mean(axis=0))
        res_reward = []


cum_reward = []
for i in range(num_take_count):
    cum_reward.append(np.mean(res_mean[0][0:i + 1]))


plt.plot(np.arange(len(cum_reward)), cum_reward, label='3 arms')
plt.xlabel('Round')
plt.ylabel('Mean reward')
plt.title("Mean reward by round and number of arms")
plt.legend()
plt.show()