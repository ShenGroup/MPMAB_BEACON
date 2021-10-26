from strategy_and_environment.strategy_banditsBEACON import BEACON_FullSensingMultiPlayerMAB
from strategy_and_environment.strategy_banditsMETC import METC_FullSensingMultiPlayerMAB
from strategy_and_environment.strategy_BEACON import BEACON
from strategy_and_environment.strategy_METC_maxmin import VaryMeanMETCElim_max_min
from strategy_and_environment.strategy_CUCB import CUCB
import numpy as np
import matplotlib.pyplot as plt
import random


random.seed(123)
legend = ['BEACON', 'METCElim', 'CUCB']
T = 1000000
K = 8
M = 6
run = 3

mean = [
      [0.45,0.49,0.59,0.17,0.37,0.86,0.94,0.98],
      [0.39,0.25,0.4,0.6,0.24,0.54,0.43,0.67],
      [0.39,0.33,0.8,0.01,0.12,0.2,0.61,0.77],
      [0.95,0.22,0.24,0.88,0.2,0.12,0.29,0.3],
      [0.69,0.89,0.25,0.59,0.43,0.18,0.01,0.84],
      [0.97,0.15,0.89,0.16,0.09,0.57,0.61,0.19]
]

agent = [BEACON_FullSensingMultiPlayerMAB(mean, M, K, T, BEACON, reward_func='max_min fairness'),
         METC_FullSensingMultiPlayerMAB(mean, M, K, T, VaryMeanMETCElim_max_min, reward_func='max_min fairness'),
         CUCB(mean, M, K, T, reward_func='max_min fairness')
         ]

col = ['blue', 'red', 'green']
ave_regret = np.zeros((3, T))

for n in range(run):
    for i in range(3):
        agent[i].simulate()
        regret = agent[i].get_results()
        ave_regret[i] += regret
        agent[i].reset()
        print(legend[i], regret[T-1])
        # np.save(legend[i] + str(n) + 'max_min.npy', regret)

plt.figure()
for i in range(3):
      ave_regret[i] /= run
      plt.plot(range(T), ave_regret[i], color=col[i])
plt.legend(legend)
plt.show()




