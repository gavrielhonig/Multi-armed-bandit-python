######################################################
# Alpha tuning - set the length of confidence interval
######################################################
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from run_file import LinUCB, offlineEvaluate

arms = np.load('arms.npy')
rewards = np.load('rewards.npy')
contexts = np.load('contexts.npy')
num_of_events = np.load('num_of_events.npy')


# First Interval [0.2 0.4 0.6 0.8 1. ]
alpha_range_one_decimal = np.linspace(0, 1, 6)
alpha_range_one_decimal = np.delete(alpha_range_one_decimal, 0)  # delete zero
# Second Interval [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 ]
alpha_range_two_decimal = np.linspace(0, 0.1, 11)
alpha_range_two_decimal = np.delete(alpha_range_two_decimal, 0)  # delete zero
# Append two intervals and sort
alpha_range = np.append(alpha_range_two_decimal, alpha_range_one_decimal)
alpha_range = np.sort(alpha_range)
results_LinUCB_with_alpha = []
#results_LinThompson_with_alpha = []

for alpha in alpha_range:
    mab = LinUCB(5, 4, alpha)
    results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts,num_of_events)[0]
    results_LinUCB_with_alpha.append(np.mean(results_LinUCB))



plt.plot(alpha_range, results_LinUCB_with_alpha,linestyle='dashed')
plt.ylabel('mean_reward')
plt.xlabel('alpha_range')
plt.title('alpha optimization')
plt.show()


print(alpha_range[results_LinUCB_with_alpha.index(max(results_LinUCB_with_alpha))])