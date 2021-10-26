from .communication_helper import *


class METC_FullSensingMultiPlayerMAB(object):
    """
    Structure of stochastic MPMAB in the full sensing model. The model can be either Homogeneous or Heterogeneous.
    """

    def __init__(self,
                 means,
                 nplayers,
                 narms,
                 horizon,
                 strategy,
                 reward_func='linear',
                 **kwargs):

        self.K = narms
        self.M = nplayers
        self.T = horizon
        self.means = np.array(means)
        self.players = [strategy(narms=self.K, **kwargs) for _ in range(nplayers)]
        self.strategy = strategy
        self.rewards = []
        self.history = []
        self.reward_func = reward_func
        self.epsilon = 0.01

    def reward_function(self, rews):
        if self.reward_func == 'linear':
            return np.sum(rews)
        elif self.reward_func == 'max_min fairness':
            return np.min(rews)
        elif self.reward_func == 'proportional fairness':
            return np.sum(np.log(self.epsilon + rews))

    def simulate_single_step_rewards(self):
        return np.random.binomial(1, self.means)

    def simulate_single_step(self, plays):
        unique, counts = np.unique(plays, return_counts=True) # compute the number of pulls per arm
        # remove the collisions
        collisions = unique[counts > 1] # arms where collisions happen
        cols = np.array([p in collisions for p in plays])  # the value is 1 if there is collision (at the arm)
        rewards = self.simulate_single_step_rewards()  # generate the statistic X_k(t)
        rews = rewards[list(range(self.M)), plays] * (1 - cols)
        return list(zip(rewards[list(range(self.M)), plays], cols)), rews


    def simulate(self):
        last_flag = False

        for t in range(self.T):
            plays = [(int)(player.play()) for player in self.players]  # plays of all players
            obs, rews = self.simulate_single_step(plays)  # observations of all players

            for i in range(self.M):
                self.players[i].update(plays[i], obs[i])  # update strategies of all player
            reward_one_round = self.reward_function(rews)
            self.rewards.append(reward_one_round)  # list of rewards

            for i in range(self.M):
                if last_flag and self.players[i].is_leader and self.players[i].phase == self.players[i].EXPLORATION:
                    leader = 0
                    for k in range(self.M):
                        if self.players[k].is_leader:
                            leader = k
                    for j in range(self.players[i].record_arm_to_explore.shape[1]):
                        tmp = 0
                        for ii in range(self.M):
                            indx = self.players[ii].relative_position - 1
                            tmp += self.means[ii, self.players[leader].record_arm_to_explore[indx, j]]
                        for l in range(2**self.players[leader].p):
                            self.history.append(tmp)
                    last_flag = False

                if self.players[i].is_leader and self.players[i].phase == self.players[i].COMMUNICATION:
                    last_flag = True

    def get_results(self):
        # Find optimal matching first
        best_choice = Oracle(self.means, self.reward_func)
        top_mean = 0
        if self.reward_func == 'linear':
            top_mean = np.sum(self.means[list(range(self.M)), best_choice])
        elif self.reward_func == 'max_min fairness':
            top_mean = np.prod(self.means[list(range(self.M)), best_choice])
        elif self.reward_func == 'proportional fairness':
            top_mean = np.sum(self.means[list(range(self.M)), best_choice]*np.log(1+self.epsilon)) + np.sum((1-self.means[list(range(self.M)), best_choice])*np.log(self.epsilon))

        best_case_reward = top_mean * np.arange(1, self.T+1)
        cumulated_reward = np.cumsum(self.rewards)
        regret = best_case_reward - cumulated_reward[:self.T]
        return regret

    def reset(self):
        self.players = [self.strategy(narms=self.K) for _ in range(self.M)]
        self.rewards = []
        self.history = []
