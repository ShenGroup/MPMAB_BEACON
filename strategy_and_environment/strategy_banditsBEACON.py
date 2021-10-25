from .communication_helper import *
import copy

class BEACON_FullSensingMultiPlayerMAB(object):
    """
    Implementation of Batched Exploration with Adaptive COmmunicatioN (BEACON) for Heterogeneous and full sensing MPMAB;
    Paper: Heterogeneous Multi-player Multi-armed Bandits: Closing the Gap and Generalization, Neurips 2021.

    means is the expectation matrix of size M x K, (nplayers, narms, horizon) = (M, K, T) in the paper;
    """

    def __init__(self, means, nplayers, narms, horizon, strategy, reward_func='linear'):
        self.K = narms
        self.M = nplayers
        self.means = np.array(means)
        self.T = horizon
        self.strategy = strategy
        self.players = [self.strategy(narms=self.K, T=self.T) for _ in range(self.M)]
        self.reward_func = reward_func
        self.t = 0
        self.rewards_record = []
        self.history = []
        self.communication_flag = False
        self.epsilon = 0.01  # parameter for proportional fairness

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
        """
        return to each player its statistics and collision indicator where "plays" is the vector of plays by the players
        rewards: The vector of X_k(t), the rewards generated iid by sampling
        rews: rewards * (1 - cols), so rews[i] = 0 if collision occurs at plays[i]
        """
        unique, counts = np.unique(plays, return_counts=True)  # compute the number of pulls per arm
        # remove the collisions
        collisions = unique[counts > 1] # arms where collisions happen
        cols = np.array([p in collisions for p in plays])  # the value is 1 if there is collision (at the arm)
        rewards = self.simulate_single_step_rewards()  # generate the rewards X_k(t)
        rews = rewards[list(range(self.M)), plays] * (1 - cols)
        return list(zip(rewards[list(range(self.M)), plays], cols)), rews

    def simulate(self):
        """
        Play the game and record the reward sequence.
        Remark: for simplicity, the communication is implemented in this Class.
        """
        while self.t < self.T:
            plays = [(int)(player.play()) for player in self.players]  # plays of all players
            obs, rews = self.simulate_single_step(plays)  # observations of all players
            for i in range(self.M):
                self.players[i].update(plays[i], obs[i])  # update strategies of all player
                if self.players[i].phase == self.players[i].COMMUNICATION:  # If communication starts
                    self.communication_flag = True
            reward_one_round = self.reward_function(rews)
            self.rewards_record.append(reward_one_round)  # list of rewards
            self.t += 1
            if self.communication_flag:
                self.communication()
                self.communication_flag = False

    def communication(self):
        """Implementation of communication."""
        # Find the index of leader
        tmp = zip([player.relative_position-1 for player in self.players], list(range(self.M)))
        relative_position_to_index = {i:j for i, j in tmp}
        leader = relative_position_to_index[0]

        # Step 1, Leader sends signals to tell followers to stop exploring.
        if not self.players[leader].flag_first_communication:
            plays = [(int)(player.arm_to_explore) for player in self.players]
            for i in range(1, self.M):
                plays[leader] = plays[relative_position_to_index[i]]  # Leader chooses follower i's arm to make a collision
                obs, rews = self.simulate_single_step(plays)
                reward_one_round = self.reward_function(rews)
                self.rewards_record.append(reward_one_round)
                self.t += 1
        # Follower updates their counters
        for i in range(1, self.M):
            follower = relative_position_to_index[i]
            self.players[follower].old_counters = copy.deepcopy(self.players[follower].counters)
            self.players[follower].counters = np.floor(np.log2(self.players[follower].npulls))

        # Step 2, Statistics communication
        # We first generate the action sequence according to BEACON. Then we sample according to the actions.
        # Finally, the communication is implemented in a centralized fashion with truncated values, which is
        #   equivalent to BEACON since the communication with indicator function information always succeeds.
        # If 1st communication round, followers send statistics with the same quantization length
        # Leader always pull arm 1 (record_communication_arm[0])
        if self.players[leader].flag_first_communication:
            comm_length = int(self.players[leader].record_counters[0][0]) # the number of bits for 1 sample mean
            # Action sequence of the leader; there are M x (K - 1) sample means
            leader_actions = str(self.players[leader].record_communication_arm[0]) * int(comm_length * self.K) * (self.M - 1)
            followers_actions = []
            # Action sequence of followers
            for i in range(1, self.M):
                follower = relative_position_to_index[i]
                bit_sequence = ""
                # the bit sequence of truncated sample means
                for j in range(self.K):
                    bit_sequence += d2b(self.players[follower].empirical_means[j], comm_length)
                # we add the actions for waiting where followers always pull their own arms (i.e., bit 0)
                tmp = str(0)*int((i-1)*self.K*comm_length) + bit_sequence + str(0) * int((self.M-i-1) * self.K * comm_length)
                tmmp = ""
                for x in tmp:
                    if x == '1':
                        tmmp += str(self.players[leader].record_communication_arm[0])
                    else:
                        tmmp += str(self.players[leader].record_communication_arm[i])
                followers_actions.append(tmmp)

            # We sample according to BEACON.
            plays = np.zeros(self.M, dtype=np.int32)
            for time in range(int((self.M-1) * self.K * comm_length)):
                plays[leader] = int(leader_actions[time])
                for i in range(1, self.M):
                    follower = relative_position_to_index[i]
                    plays[follower] = int(followers_actions[i-1][time])
                obs, rews = self.simulate_single_step(plays)  # observations of all players
                reward_one_round = self.reward_function(rews)
                self.rewards_record.append(reward_one_round)  # list of rewards
                self.t += 1
            # The communication is implemented directly with truncated values, which is equivalent to BEACON.
            for i in range(1, self.M):
                follower = relative_position_to_index[i]
                for j in range(self.K):
                    self.players[leader].record_player_stat[i][j] = truncate(self.players[follower].empirical_means[j], comm_length)
        # If it is not the first time to communicate
        # In this case, followers will send the sample means (by ADC, so it is the difference actually) if the counters
        #   increase.
        else:
            leader_actions = ""
            collision_actions = ""
            Lp = 7
            # Leader will make a collision to tell follower i that leader will communicate with him.
            # Then, leader pulls his arm Lp times to receive statistics.
            for i in self.players[leader].need_communication:
                collision_actions += str(self.players[leader].record_communication_arm[i])
            tmp = str(self.players[leader].record_communication_arm[0]) * Lp
            for action in collision_actions:
                leader_actions = leader_actions + action + tmp

            total_len = len(leader_actions)
            followers_actions = []
            for i in range(1, self.M):
                follower = relative_position_to_index[i]
                count = 0
                if i in self.players[leader].need_communication:
                    arm = self.players[follower].arm_to_explore
                    # We generate the action sequence by the truncated difference, the last 1/0 bit is for the signal
                    if truncate(self.players[follower].empirical_means[arm], int(self.players[follower].counters[arm])) \
                            - truncate(self.players[follower].last_empirical_means[arm], int(self.players[follower].old_counters[arm])) > 0:
                        tmp = d2b(truncate(self.players[follower].empirical_means[arm], int(self.players[follower].counters[arm]))
                                  - truncate(self.players[follower].last_empirical_means[arm], int(self.players[follower].old_counters[arm])),
                                  int(self.players[follower].counters[arm])) + '1'
                    else:
                        tmp = d2b(truncate(self.players[follower].last_empirical_means[arm], int(self.players[follower].old_counters[arm]))
                                  - truncate(self.players[follower].empirical_means[arm], int(self.players[follower].counters[arm])),
                                  int(self.players[follower].counters[arm])) + '0'
                    bit_sequence = str(0) * int((Lp+1) * count+1) + tmp[int(self.players[follower].counters[arm]-Lp+1):] + str(0) * int(total_len-(Lp+1)*count - Lp - 1)
                    count += 1
                else:
                    # this follower does not need to send statistic so she always pulls her own arm
                    bit_sequence = str(0) * total_len
                tmmp = ""
                for x in bit_sequence:
                    if x == '1':
                        tmmp += str(self.players[leader].record_communication_arm[0])
                    else:
                        tmmp += str(self.players[leader].record_communication_arm[i])
                followers_actions.append(tmmp)

            # We sample according to BEACON.
            plays = np.zeros(self.M, dtype=np.int32)
            for time in range(total_len):
                plays[leader] = int(leader_actions[time])
                for i in range(1, self.M):
                    follower = relative_position_to_index[i]
                    plays[follower] = int(followers_actions[i-1][time])
                obs, rews = self.simulate_single_step(plays)  # observations of all players
                reward_one_round = self.reward_function(rews)
                self.rewards_record.append(reward_one_round)  # list of rewards
                self.t += 1
            # The communication is implemented directly with truncated values, which is equivalent to BEACON.
            for i in range(1, self.M):
                follower = relative_position_to_index[i]
                arm = self.players[follower].arm_to_explore
                if truncate(self.players[follower].empirical_means[arm], int(self.players[follower].counters[arm])) - truncate(self.players[follower].last_empirical_means[arm], int(self.players[follower].old_counters[arm])) > 0:
                    self.players[leader].record_player_stat[i][arm] = truncate(truncate(self.players[follower].empirical_means[arm], int(self.players[follower].counters[arm])) - truncate(self.players[follower].last_empirical_means[arm], int(self.players[follower].old_counters[arm])), Lp-1)
                else:
                    self.players[leader].record_player_stat[i][arm] = -truncate(truncate(self.players[follower].last_empirical_means[arm], int(self.players[follower].old_counters[arm]))-truncate(self.players[follower].empirical_means[arm], int(self.players[follower].counters[arm])), Lp-1)

        # Leader updates UCB and finds the matching to explore in the next exploration phase
        self.players[leader].record_player_stat += self.players[leader].record_last_phase_player_stat
        self.players[leader].record_player_stat[0] = self.players[leader].empirical_means
        for i in range(self.M):
            for j in range(self.K):
                self.players[leader].record_UCB[i][j] = self.players[leader].record_player_stat[i][j] + np.sqrt(3*np.log(self.t)/(2**(self.players[leader].record_counters[i][j]+1)))
        # Find the matching to explore and compute the exploration length
        self.players[leader].record_matching_to_explore = Oracle(self.players[leader].record_UCB)
        tmp = self.players[leader].record_counters[list(range(self.M)), self.players[leader].record_matching_to_explore]
        self.players[leader].explore_length = 2**np.min(tmp)
        # Step 3: leader sends the arm that player j needs to explore in the next exploration phase to player j
        leader_actions = ""
        plays = np.zeros(self.M, dtype=np.int32)
        for i in range(1, self.M):
            tmp = bin(self.players[leader].record_matching_to_explore[i])[2:]
            leader_actions += str(self.players[leader].record_communication_arm[0]) * (self.players[leader].commArmLength - len(tmp)) + tmp
            plays[relative_position_to_index[i]] = self.players[leader].record_communication_arm[i]
        for time in range((self.M-1) * self.players[leader].commArmLength):
            plays[leader] = leader_actions[time]
            obs, rews = self.simulate_single_step(plays)
            reward_one_round = self.reward_function(rews)
            self.rewards_record.append(reward_one_round)
            self.t += 1
        # At the end of communication, players update many things
        self.players[leader].state_round = 0
        self.players[leader].phase = self.players[leader].EXPLORATION
        self.players[leader].arm_to_explore = self.players[leader].record_matching_to_explore[0]
        self.players[leader].record_last_phase_player_stat = copy.deepcopy(self.players[leader].record_player_stat)
        self.players[leader].record_player_stat = np.zeros((self.M, self.K))
        self.players[leader].record_old_counters = copy.deepcopy(self.players[leader].record_counters)
        self.players[leader].record_communication_arm = self.players[leader].record_matching_to_explore
        for i in range(1, self.M):
            follower = relative_position_to_index[i]
            self.players[follower].arm_to_explore = self.players[leader].record_matching_to_explore[i]
            self.players[follower].state_round = 0
            self.players[follower].phase = self.players[follower].EXPLORATION
            self.players[follower].last_empirical_means = copy.deepcopy(self.players[follower].empirical_means)
            self.players[follower].communication_arm = self.players[leader].record_communication_arm[i]
            self.players[follower].leader_arm = self.players[leader].record_communication_arm[0]

        if self.players[leader].flag_first_communication:
            self.players[leader].flag_first_communication = False

        #################################################################################
        # Record the play history for plot
        #tmp = 0
        #for i in range(self.M):
        #    xx = self.players[i].relative_position - 1
        #    tmp += self.means[i, self.players[leader].record_matching_to_explore[xx]]
        #for k in range(int(self.players[leader].explore_length)):
        #    self.history.append(tmp)
        #################################################################################


    def get_results(self):
        #ratio_history = np.zeros((self.M, self.K))
        #for i in range(self.M):
        #    for j in range(self.K):
        #        ratio_history[i][j] = self.players[i].npulls[j]
        #ratio_history /= np.sum(ratio_history)

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
        # Compute the regret
        cumulated_reward = np.cumsum(self.rewards_record)
        regret = best_case_reward - cumulated_reward[:self.T]
        return regret[:self.T]

    def reset(self):
        self.players = [self.strategy(narms=self.K, T=self.T) for _ in range(self.M)]
        self.t = 0
        self.rewards_record = []
        self.history = []
        self.communication_flag = False
