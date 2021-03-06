from strategy_and_environment.strategy_banditsBEACON import BEACON_FullSensingMultiPlayerMAB
from strategy_and_environment.strategy_banditsMETC import METC_FullSensingMultiPlayerMAB
from strategy_and_environment.strategy_METC import VaryMeanMETCElim
from strategy_and_environment.strategy_BEACON import BEACON
import numpy as np
import matplotlib.pyplot as plt
import random


random.seed(123)
reward_function = 'linear'
legend = ['BEACON', 'METCElim']
T = 1000000
K = 5
M = 5
run = 15  # Test on 100 instances
result = np.zeros((2, run))
mean_regret = np.zeros(2)

for t in range(run):
    mean = np.random.random((M, K))
    # print(t, mean)
    print(t)
    agent = [BEACON_FullSensingMultiPlayerMAB(mean, M, K, T, BEACON, reward_func=reward_function),
         METC_FullSensingMultiPlayerMAB(mean, M, K, T, VaryMeanMETCElim, reward_func=reward_function),
         ]
    for j in range(2):
        agent[j].simulate()
        tmp = agent[j].get_results()
        mean_regret[j] += tmp[T-1]
        print(legend[j], tmp[T-1])
        result[j, t] = tmp[T-1]

for j in range(2):
    print(mean_regret[j] / run)

# np.save('random_BEACON_pr.npy', result[0, :])
# np.save('random_METC_pr.npy', result[1, :])

plt.figure()
plt.hist(result[0,:], bins=20)
plt.hist(result[1,:], bins=20)
plt.show()
