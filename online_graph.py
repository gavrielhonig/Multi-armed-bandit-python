#################################################
# Run online simulation on Lightricks data.
# Sample from bernoulli distributions with
# different probabilities.
# No loss of data comparing to offline data.
#################################################

import numpy as np
import pandas as pd
from run_file import LinUCB,UCB,GB,GreedyEpsilon
from run_file import offlineEvaluate,offlineEvaluate_GB,onlineEvaluate
import matplotlib.pyplot as plt

arms = np.load('arms.npy')
rewards = np.load('rewards.npy')
contexts = np.load('contexts.npy')
num_of_events = np.load('num_of_events.npy')

mab_linUCB = LinUCB(5, contexts.shape[1], 0.1)
mab_UCB = UCB(5, 1)
#mab_GB = GB(5,4)
mab_greedy = GreedyEpsilon(5,0.2)

results_LinUCB, arms_chosen_LinUCB = onlineEvaluate(mab_linUCB, contexts, 0.3,0.6,num_of_events)
print(2)
results_UCB, arms_chosen_UCB = onlineEvaluate(mab_UCB, arms, 0.3,0.6, num_of_events)
print(3)
results_greedy, arms_chosen_greedy = onlineEvaluate(mab_greedy, arms,0.3,0.6, num_of_events)
#results_GB, arms_chosen_GB = offlineEvaluate_GB(mab_GB, arms, rewards, contexts, num_of_events)

dict_linUCB = {'arm_LUCB':arms_chosen_LinUCB, 'reward_LUCB': results_LinUCB}
dict_UCB = {'arm_UCB': arms_chosen_UCB, 'reward_UCB': results_UCB}
dict_greedy = {'arm_greedy': arms_chosen_greedy, 'reward_greedy': results_greedy}
#dict_GB = {'arm_GB': arms_chosen_GB, 'reward_GB': results_GB}

df_linUCB = pd.DataFrame(dict_linUCB)
df_UCB = pd.DataFrame(dict_UCB)
df_greedy = pd.DataFrame(dict_greedy)
#df_GB = pd.DataFrame(dict_GB)

cum_reward_LUCB = []
for i in range(df_linUCB.shape[0]):
    cum_reward_LUCB.append(np.mean(df_linUCB['reward_LUCB'][0:i+1]))

cum_reward_UCB = []
for i in range(df_UCB.shape[0]):
    cum_reward_UCB.append(np.mean(df_UCB['reward_UCB'][0:i+1]))

cum_reward_greedy = []
for i in range(df_greedy.shape[0]):
    cum_reward_greedy.append(np.mean(df_greedy['reward_greedy'][0:i+1]))

# cum_reward_GB = []
# for i in range(df_GB.shape[0]):
#     cum_reward_GB.append(np.mean(df_GB['reward_GB'][0:i+1]))

plt.plot(df_linUCB.index,cum_reward_LUCB, label='LinUCB')
plt.plot(df_UCB.index,cum_reward_UCB, label='UCB')
plt.plot(df_greedy.index,cum_reward_greedy, label='Epsilon Greedy')
#plt.plot(df_GB.index,cum_reward_GB, label='Greedy Bandit')

plt.xlabel('Rounds')
plt.ylabel('Per-Round Cumulative Reward')
plt.ylim((0.4, 0.65))
plt.title('Mean Reward by round- All Algorithms')
plt.legend()
plt.show()

