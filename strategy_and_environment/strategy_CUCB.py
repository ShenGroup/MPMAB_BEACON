from .communication_helper import *


class CUCB(object):
    """
    CUCB is from "Combinatorial multi-armed bandit: General framework and applications", ICML201
    The algorithm maintains a table of UCBs and finds the optimal matching by this UCB matrix.
    """
    def __init__(self, means, nplayers, narms, horizon, reward_func='linear'):
        self.K = narms
        self.M = nplayers
        self.means = np.array(means)
        self.reward_func = reward_func
        self.t = 0
        self.T = horizon
        self.empirical_means = np.zeros((self.M, self.K))
        self.npulls = np.zeros((self.M, self.K), dtype=np.int32)
        self.UCB = np.zeros((self.M, self.K))
        self.rewards_record = []
        self.epsilon = 0.01

    def reward_function(self, rews):
        if self.reward_func == 'linear':
            return np.sum(rews)
        elif self.reward_func == 'max_min fairness':
            return np.min(rews)
        elif self.reward_func == 'proportional fairness':
            return np.sum(np.log(self.epsilon + rews))

    def simulate_single_step_rewards(self, plays):
        return np.random.binomial(1, self.means)[list(range(self.M)), plays]

    def _update(self, plays, rews):
        for i in range(self.M):
            self.empirical_means[i][plays[i]] = (self.empirical_means[i][plays[i]] * self.npulls[i][plays[i]] + rews[i]) / (self.npulls[i][plays[i]] + 1)
            self.npulls[i][plays[i]] += 1
        for i in range(self.M):
            for j in range(self.K):
                self.UCB[i][j] = self.empirical_means[i][j] + np.sqrt(3 * np.log(self.t) / (2 * self.npulls[i][j]))

    def simulate(self):
        for k in range(self.K):
            plays = np.ones(self.M, dtype=np.int32) * k
            rews = self.simulate_single_step_rewards(plays)  # observations of all players
            reward_one_round = self.reward_function(rews)
            self.rewards_record.append(reward_one_round)  # list of rewards
            for i in range(self.M):
                self.empirical_means[i][plays[i]] = (self.empirical_means[i][plays[i]] * self.npulls[i][plays[i]] + rews[i]) /  (self.npulls[i][plays[i]] + 1)
                self.npulls[i][plays[i]] += 1
            self.t += 1

        for i in range(self.M):
            for j in range(self.K):
                self.UCB[i][j] = self.empirical_means[i][j] + np.sqrt(3 * np.log(self.t) / (2 * self.npulls[i][j]))

        while self.t < self.T:
            if self.t % 10000 == 0:
                print("Time step: ", self.t)
            plays = Oracle(self.UCB)
            rews = self.simulate_single_step_rewards(plays)  # observations of all players
            reward_one_round = self.reward_function(rews)
            self.rewards_record.append(reward_one_round)  # list of rewards
            self._update(plays, rews)
            self.t += 1

    def get_results(self):
        # Find optimal matching first
        best_choice = Oracle(self.means,self.reward_func)
        top_mean = 0
        if self.reward_func == 'linear':
            top_mean = np.sum(self.means[list(range(self.M)), best_choice])
        elif self.reward_func == 'max_min fairness':
            top_mean = np.prod(self.means[list(range(self.M)), best_choice])
        elif self.reward_func == 'proportional fairness':
            top_mean = np.sum(self.means[list(range(self.M)), best_choice]*np.log(1+self.epsilon)) + np.sum((1-self.means[list(range(self.M)), best_choice])*np.log(self.epsilon))

        best_case_reward = top_mean * np.arange(1, self.T+1)
        cumulated_reward = np.cumsum(self.rewards_record)
        regret = best_case_reward - cumulated_reward[:self.T]

        return regret[:self.T]

    def reset(self):
        self.t = 0
        self.empirical_means = np.zeros((self.M, self.K))
        self.npulls = np.zeros((self.M, self.K), dtype=np.int32)
        self.UCB = np.zeros((self.M, self.K))
        self.rewards_record = []
