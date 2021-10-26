from strategy_and_environment.strategy_banditsBEACON import BEACON_FullSensingMultiPlayerMAB
from strategy_and_environment.strategy_banditsMETC import METC_FullSensingMultiPlayerMAB
from strategy_and_environment.strategy_METC import VaryMeanMETCElim
from strategy_and_environment.strategy_BEACON import BEACON
from strategy_and_environment.strategy_CUCB import CUCB
import numpy as np
import matplotlib.pyplot as plt
import random


random.seed(123)
legend = ['BEACON', 'METCElim', 'CUCB']
T = 1000000
K = 5
M = 5
run = 3
mean = np.array([[0.5, 0.49, 0.39, 0.29, 0.5],
      [0.5, 0.49, 0.39, 0.29, 0.19],
      [0.29, 0.19, 0.5, 0.499, 0.39],
      [0.29, 0.49, 0.5, 0.5, 0.39],
      [0.49, 0.49, 0.49, 0.49, 0.5]])

agent = [BEACON_FullSensingMultiPlayerMAB(mean, M, K, T, BEACON, reward_func='linear'),
         METC_FullSensingMultiPlayerMAB(mean, M, K, T, VaryMeanMETCElim, reward_func='linear'),
         CUCB(mean, M, K, T, reward_func='linear')
         ]
col = ['blue', 'red', 'green']
ave_regret = np.zeros((3, T))

for n in range(run):
      for i in range(3):
            agent[i].simulate()
            regret = agent[i].get_results()
            ave_regret[i] += regret
            agent[i].reset()
            print(n+1, legend[i], regret[T-1])

plt.figure()
for i in range(3):
      ave_regret[i] /= run
      plt.plot(range(T), ave_regret[i], color=col[i])
plt.legend(legend)
plt.show()


