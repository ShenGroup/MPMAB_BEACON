import numpy as np
from scipy.optimize import  linear_sum_assignment

class PlayerStrategy():

    def __init__(self, narms, T):
        self.T = T # horizon
        self.t = 0 # current round
        self.K = narms # true number of arms (Kp used as number of active arms)

class VaryMeanMETCElim(PlayerStrategy):
    """
    Implementation of METCElim algorithm introduced by the paper
    "A practical algorithm for multiplayer bandits when arm means vary among players", AISTATS2020.
    """

    def __init__(self, narms, T=100, c=1, verbose=False):
        PlayerStrategy.__init__(self, narms, T)
        self.M = 0 # (estimated) number of players

        # phase
        # In this program, all players start with INIT_.... Then, they alternate between EXPLORATION and COMMUNICATION
        # When there is only one matching left, all players enter EXPLOIT phase.
        self.INIT_ORTHOG_SAMPLE = 1
        self.INIT_ORTHOG_VERIFICATION = 2
        self.INIT_RANK_ASSIGN = 3
        self.EXPLORATION = 4
        self.EXPLOIT = 10
        self.COMMUNICATION = 11
        self.phase_names = {
            self.INIT_ORTHOG_SAMPLE: 'INIT_ORTHOG_SAMPLE',
            self.INIT_ORTHOG_VERIFICATION: 'INIT_ORTHOG_VERIFICATION',
            self.INIT_RANK_ASSIGN: 'INIT_RANK_ASSIGN',
            self.EXPLORATION: 'EXPLORATION',
            self.EXPLOIT: 'EXPLOIT',
            self.COMMUNICATION: 'COMMUNICATION',
        }
        # variables for all phases
        self.state_round = 1 # this var is used for tracking time steps within one phase
        self.phase = self.INIT_ORTHOG_SAMPLE
        self.p = 5 # we start with p > 1 to accelerate exploration
        self.initial_p = self.p
        self.verbose = verbose # used to debug, it makes code run slowly if verbose == true
        self.c = c
        # variables used for initialization
        self.id = 0 # 0 until fixed on some arm
        self.i = 0
        self.collisions = 0
        # variables for exploration
        self.cumulative_rewards = np.zeros(self.K) # empirical means
        self.npulls = np.zeros(self.K, dtype=np.int32) # number of pulls for each arm
        self.empirical_means = np.zeros(self.K)

        self.Tp = 0 # used for calculating Qp and confidence radius
        self.arm_to_explore = np.array(range(self.K), dtype=np.int32)
        # at the end of communication phase, players set last_empirical_means = empirical_means

        # variables used for communication
        self.Qp = 0 # calculated at the beginning of communication phase
        self.commArmLength = int(np.ceil(np.log2(self.K)))
        self.commEpLength = 0
        self.communication_arm = None
        self.leader_arm = 0

        self.new_arm_to_explore = np.array([[]], dtype=np.int32)
        self.new_leader_arm = 0
        self.new_communication_arm = 0
        self.Ep_number = 0 # length of Ep

        # state varaiable
        self.is_active = True
        self.exploited_arm = -1  # Taking value in {-1,1,2,3,...,K}
        self.is_leader = False

        # variables for leader
        self.record_arm_empirical_means = None
        self.record_player_stat = None
        self.record_player_npulls = None # since players of exploitation phase keep communicating
        self.record_arm_to_explore = None
        self.active_matching = None
        self.commArms = None
        self.INF = 88888
        self.flag = None
        self.flag2 = None
        # debug
        if self.verbose == 1:
            self.log_func = lambda x: print(x)
        else:
            self.log_func = lambda x: self.log.append(x)
        self.log = []  # Logging feature
        self.log_counter = 0  # Log id

    def _update_arm_stats(self, arm, reward, col):
        '''
        update statistics
        arm: chosen arm in {1,2,...,K}
        reward: X_arm(t)
        col: indicator of collision
        '''
        self.cumulative_rewards[arm-1] += reward * (1-col)
        self.npulls[arm-1] += 1

    def _log_phase(self):
        self._add_log('Phase is {}'.format(self.phase_names[self.phase]))

    def _log_id(self):
        self._add_log('Player id: {}'.format(self.id))

    def _add_log(self, text):
        if self.verbose > 0:
            self.log_func('[{}] [{}] - {}'.format(self.phase, self.state_round, text))
            self.log_counter += 1

    def _add_log_by_phase(self, text):
        if self.verbose > 0:
            self.log_func('[{}] - {}'.format(self.p, text))
            self.log_counter += 1

    def _add_log_by_round(self, text):
        if self.verbose > 0:
            self.log_func('[{}] - {}'.format(self.t, text))
            self.log_counter += 1

    def play(self):
        '''
        The strategy for a player to decide which arm to play
        '''
        a = -1
        # In sampling we choose an arm uniformly if the ID is 0, otherwise
        # we choose id.
        if self.phase == self.INIT_ORTHOG_SAMPLE:
            # We don't sample the k-th arm, reserved for verification
            a = self.id if self.id > 0 else np.random.choice(  # select one arm randomly from {1,2,...,K-1}
                [i + 1 for i in range(self.K - 1)])
        # In verification, depending on the round and our ID, we choose our
        # ID or arm K.
        elif self.phase == self.INIT_ORTHOG_VERIFICATION: # K rounds
            if self.state_round != self.id and self.id > 0:
                a = self.id
            else:
                a = self.K
        # The goal of this phase is to understand how many Players
        # there are, our relative position, and if we are the leader.
        elif self.phase == self.INIT_RANK_ASSIGN: # K-1 blocks
            a = self.id + max(0, self.i) if (
                self.id + self.i <= self.K) else self.id
            self.i += 1 if (self.state_round >= 2 * self.id - 1) else 0
        elif self.phase == self.EXPLORATION:
            if self.p == self.initial_p:
                a = self.arm_to_explore[(self.relative_position - 1 + self.state_round) % self.K] + 1
            else:
                a = self.arm_to_explore[self.state_round % len(self.arm_to_explore)] + 1

        elif self.phase == self.COMMUNICATION:
            if self.is_leader:
                a = self.leader_comm() + 1
            else:
                a = self.follower_comm() + 1
        elif self.phase == self.EXPLOIT:
            a = self.exploited_arm + 1

        if a == -1:
            self._add_log_by_phase(str(self.relative_position) + " Player cannot decide arm to play")
        if a is None:
            print(str(self.relative_position) + " ", self.p)

        return a - 1

    def update(self, play, obs):
        '''
        play: {0,1,...,K-1}
        '''
        self.t += 1
        arm = play + 1
        rewards, col = obs
        # If we are in Initialization.Sampling state
        if self.phase == self.INIT_ORTHOG_SAMPLE:
            self.phase = self.INIT_ORTHOG_VERIFICATION
            self.collisions = 0  # used to count collisions and thus to estimate M
            self.state_round = 1
            if self.id == 0:
                self.id = arm if col == 0 else 0
        # If we are in Initialization.Verification state
        elif self.phase == self.INIT_ORTHOG_VERIFICATION:
            self.collisions += col
            if self.state_round == self.K:
                if self.collisions == 0:
                    self.phase = self.INIT_RANK_ASSIGN
                    self.state_round = 1
                    self.collisions = 0
                else: # restart sample phase
                    self.phase = self.INIT_ORTHOG_SAMPLE
                    self.state_round = 1
                    self.collisions = 0
            else:
                self.state_round += 1
        elif self.phase == self.INIT_RANK_ASSIGN:
            # Increase number of collisions, this will be equal to M at the end of
            # the round.
            self.collisions += col
            if self.state_round == 2 * self.id - 1:
                self.relative_position = self.collisions + 1
                self._add_log('Identified relative position: {}'.format(
                    self.relative_position))
                if self.collisions == 0:
                    self.is_leader = True
                    self._add_log('We are leader')

            self.state_round += 1
            # Finished all the rounds, move to next phase
            if self.state_round == 2 * self.K - 2:
                # At the end of our block we check how many collisions we
                # had. That number is the number of players
                self.M = self.collisions + 1
                self.Mp = self.M
                self.collisions = 0
                self.active_players = np.array(range(self.M), dtype=np.int32) + 1
                self.communication_arm = self.relative_position - 1
                self.commEpLength = int(np.ceil(np.log2(self.M * self.K)))

                if self.is_leader: # leader has to store many things
                    self.record_player_stat = np.zeros((self.M, self.K))
                    self.record_player_npulls = np.zeros((self.M, self.K))
                    self.flag = np.zeros((self.M, self.K), dtype=np.int32)
                    self.flag2 = np.zeros((self.M, self.K), dtype=np.int32)
                    self.record_player_exploited_arms = np.zeros(self.M)
                    self.record_arm_to_explore = np.zeros((self.M, self.K), dtype=np.int32)
                    self.record_arm_empirical_means = np.zeros((self.M, self.K))
                    self.active_matching = np.ones((self.M, self.K), dtype=np.int32)
                    self.commArms = np.array(range(self.M), dtype=np.int32)
                    for i in range(self.M):
                        for j in range(self.K):
                            self.record_arm_to_explore[i][j] = j
                # i and state_round are used as a clock within one phase
                self.state_round = 0
                self.phase = self.EXPLORATION
                return
        elif self.phase == self.EXPLORATION:
            self._update_arm_stats(arm, rewards, col)
            self.state_round += 1
            # end of exploration
            if self.state_round == len(self.arm_to_explore) * np.ceil(2**(self.p**self.c)):
                self.phase = self.COMMUNICATION
                self.state_round = 0
                self.empirical_means = self.cumulative_rewards / self.npulls
                self.Tp += int(np.ceil(2**(self.p**self.c)))
                epsilonp = np.sqrt(np.log(self.M**2*self.K*self.T)/2/self.Tp)
                self.Qp = int(np.ceil(-np.log10(0.1*epsilonp)))
                if self.is_leader:
                    for i in range(1, self.M):
                        for j in self.record_arm_to_explore[i]:
                            self.record_player_npulls[i][j] += np.ceil(2**(self.p**self.c))

        elif self.phase == self.COMMUNICATION:
            if self.is_leader:
                self.leader_comm_update(play, obs)
            else:
                self.follower_comm_update(play, obs)

        elif self.phase == self.EXPLOIT:
            return

    def truncate(self, x, Q):
        bit_representation = self.d2b(x, Q)
        y = 0
        for q in range(Q):
            if bit_representation[q] == '1':
                y += 2 ** (-q - 1)
        return y

    def d2b(self, decimal, q):
        '''
        This function return the bit sequence of decimal which is truncated by q bits
        decimal: [0,1], positive
        q: sequence length
        '''
        i = 0
        if decimal == 1:
            return str(1).rjust(q, '1')
        decimal_convert = ""
        while decimal != 0 and i < q:
            result = int(decimal * 2)
            decimal = decimal * 2 - result
            decimal_convert = decimal_convert + str(result)
            i = i + 1
        while i < q:
            decimal_convert = decimal_convert + "0"
            i = i + 1
        return decimal_convert

    def get_optimal_matching(self, pair=None):
        # return the empirically optimal matching if pair == None
        # return the empirically optimal matching including pair [player, arm] if pair != None
        self.record_arm_empirical_means = -self.record_arm_empirical_means
        if pair == None:
            rind, cind = linear_sum_assignment(self.record_arm_empirical_means)
        else:
            temp = self.record_arm_empirical_means[pair[0], pair[1]]
            self.record_arm_empirical_means[pair[0], pair[1]] = -self.INF
            rind, cind = linear_sum_assignment(self.record_arm_empirical_means)
            self.record_arm_empirical_means[pair[0], pair[1]] = temp
        self.record_arm_empirical_means = -self.record_arm_empirical_means
        return cind

    def leader_update_active_set(self):
        # leader update many things after receiving stat
        # First, leader find pi*
        pi_star = self.get_optimal_matching()
        self.optimal_matching = pi_star
        # Second, for each active pair, we do something
        explore_matching = []
        explore_matching.append(list(pi_star))
        flag = np.zeros((self.M, self.K))
        for i in range(self.M):
            flag[i][pi_star[i]] = 1
        epsilonp = np.sqrt(np.log(self.M**2*self.K*self.T)/2/self.Tp)
        for i in range(self.M):
            for j in range(self.K):
                if self.active_matching[i][j] == 1:
                    # find the empirically optimal matching for this pair
                    hat_pi = self.get_optimal_matching(pair=[i,j])
                    if np.sum(self.record_arm_empirical_means[list(range(self.M)), pi_star]-self.record_arm_empirical_means[list(range(self.M)), hat_pi]) <= 2.2*self.M*epsilonp: #### The condition in paper
                        if flag[i][j] == 1:
                            continue
                        if not(list(hat_pi) in explore_matching):
                            explore_matching.append(list(hat_pi))
                            for k in range(self.M):
                                flag[k][hat_pi[k]] = 1
                    else:
                        self.active_matching[i][j] = 0
                        self.record_arm_empirical_means[i][j] = -self.INF
        self.new_arm_to_explore = np.transpose(np.array(explore_matching))

    def leader_comm(self):
        # first time to communicate
        if self.state_round < (self.M-1) * self.record_arm_to_explore.shape[1] * self.Qp:  # receive statistics
            return self.commArms[0]
        T0 = (self.M-1) * self.record_arm_to_explore.shape[1] * self.Qp

        if self.state_round < T0 + (self.M-1) * self.commArmLength*2:
            k0 = (self.state_round - T0) // (self.commArmLength*2) + 1
            a = self.commArms[k0]
            t0 = int(self.state_round - T0 - (k0-1) * self.commArmLength*2)
            index1 = t0 // self.commArmLength
            index2 = t0 % self.commArmLength
            if index1 == 0:
                if(self.optimal_matching[0] >> index2) % 2:
                    return a
                else:
                    return self.commArms[0]
            else:
                if(self.optimal_matching[k0] >> index2) % 2:
                    return a
                else:
                    return self.commArms[0]

        T1 = T0 + (self.M-1) * self.commArmLength*2

        if self.state_round < T1 + (self.M-1) * self.commEpLength:
            k0 = (self.state_round - T1) // self.commEpLength + 1
            a = self.commArms[k0]
            t0 = int(self.state_round - T1 - (k0-1) * self.commEpLength)
            if(self.new_arm_to_explore.shape[1] >> t0) % 2:
                return a
            else:
                return self.commArms[0]
        T2 = T1 + (self.M-1) * self.commEpLength
        if self.state_round < T2 + (self.M-1) * self.commArmLength * self.new_arm_to_explore.shape[1]:
            k0 = (self.state_round - T2) // (self.commArmLength * self.new_arm_to_explore.shape[1]) + 1
            a = self.commArms[k0]
            t0 = int(self.state_round - T2 - (k0-1)*self.commArmLength * self.new_arm_to_explore.shape[1])
            index1 = t0 // self.commArmLength
            index2 = t0 % self.commArmLength
            if (self.new_arm_to_explore[k0][index1] >> index2) % 2:
                return a
            else:
                return self.commArms[0]

    def follower_comm(self):
        # first time to communicate
        if True:
            if self.state_round < (self.M-1) * len(self.arm_to_explore) * self.Qp:  # receive statistics
                k0 = self.state_round // (len(self.arm_to_explore) * self.Qp) + 1  # player to receive stat
                if k0 == (self.relative_position - 1):
                    t0 = self.state_round - (self.relative_position - 2) * (len(self.arm_to_explore) * self.Qp)
                    index1 = self.arm_to_explore[t0 // self.Qp]
                    index2 = t0 % self.Qp
                    bit_sequence = self.d2b(self.empirical_means[index1], self.Qp)

                    if bit_sequence[index2] == '1':
                        return self.leader_arm
                    else:
                        return self.communication_arm
                else:
                    return self.communication_arm
            else:
                return self.communication_arm

    def leader_comm_update(self, play, obs):
        arm = play + 1
        rewards, col = obs
        # first time to communicate
        if True:
            # shall be Qp
            if self.state_round < (self.M-1) * self.record_arm_to_explore.shape[1] * self.Qp:  # receive statistics
                k0 = self.state_round // (self.record_arm_to_explore.shape[1] * self.Qp) + 1  # player to receive stat
                t0 = self.state_round - (k0 - 1) * (self.record_arm_to_explore.shape[1] * self.Qp)
                index1 = self.record_arm_to_explore[k0][t0 // self.Qp]
                index2 = t0 % self.Qp + 1
                if self.flag[k0][index1] == 0:
                    if index2 == self.Qp:
                        self.flag[k0][index1] = 1
                    if col:
                        self.record_player_stat[k0][index1] += 2**(-index2)
                    else:
                        pass
            T0 = (self.M-1) * self.record_arm_to_explore.shape[1] * self.Qp
            if self.state_round == T0 - 1: # After receiving statistics, leader updates many things

                for i in range(1, self.M):
                    for j in range(self.record_arm_to_explore.shape[1]):
                        index = self.record_arm_to_explore[i][j]
                        if self.flag2[i][index] == 0:
                            self.record_arm_empirical_means[i][index] = self.record_player_stat[i][index]
                            self.flag2[i][index] = 1

                for j in range(self.record_arm_to_explore.shape[1]):
                    index = self.record_arm_to_explore[0][j]
                    if self.flag2[0][index] == 0:
                        self.record_arm_empirical_means[0][index] = self.cumulative_rewards[index] / self.npulls[index]
                        self.flag2[0][index] = 1
                self.leader_update_active_set()

            T1 = T0 + (self.M-1) * self.commArmLength*2
            if self.state_round == T1 - 1:
                self.commArms = self.optimal_matching
            T2 = T1 + (self.M-1) * self.commEpLength

            self.state_round = self.state_round + 1

            if self.state_round == T0 + (self.M-1) * self.commArmLength*2 + (self.M-1) * self.commEpLength + (self.M-1) * self.commArmLength * self.new_arm_to_explore.shape[1]:
                # end of communication phase
                self._add_log_by_round('Leader ends comm')
                self.record_arm_to_explore = self.new_arm_to_explore
                self.arm_to_explore = self.record_arm_to_explore[0]
                self.phase = self.EXPLORATION
                self.state_round = 0
                self.record_player_stat = np.zeros((self.M, self.K))
                self.p = self.p + 1
                self.flag = np.zeros((self.M, self.K), dtype=np.int32)
                self.flag2 = np.zeros((self.M, self.K), dtype=np.int32)

                if self.record_arm_to_explore.shape[1] == 1:
                    self.phase = self.EXPLOIT
                    self.exploited_arm = self.arm_to_explore[0]
                return

    def follower_comm_update(self, play, obs):
        arm = play + 1
        rewards, col = obs

        T0 = (self.M-1) * len(self.arm_to_explore) * self.Qp

        if T0 <= self.state_round < T0 + (self.M-1) * self.commArmLength*2:
            k0 = (self.state_round - T0) // (self.commArmLength*2) + 1  # player to com
            t0 = int(self.state_round - T0 - (k0-1) * self.commArmLength*2)
            if self.relative_position == (k0+1):
                index1 = t0 // self.commArmLength
                index2 = t0 % self.commArmLength
                if index1 == 0:
                    if col:
                        self.new_leader_arm += 2**index2
                else:
                    if col:
                        self.new_communication_arm += 2**index2
            else:
                pass
        T1 = T0 + (self.M-1) * self.commArmLength*2
        if self.state_round == T1 -1:
            self.leader_arm = self.new_leader_arm
            self.communication_arm = self.new_communication_arm

        if T1 <= self.state_round < T1 + (self.M-1) * self.commEpLength:
            k0 = (self.state_round - T1) // self.commEpLength + 1  # player to com
            t0 = int(self.state_round - T1 - (k0-1) * self.commEpLength)
            if self.relative_position == (k0+1):
                if col:
                    self.Ep_number = self.Ep_number + 2**t0
                else:
                    pass
        T2 = T1 + (self.M-1) * self.commEpLength
        if self.state_round == T2 - 1:
            self.new_arm_to_explore = np.zeros(self.Ep_number, dtype=np.int32)

        if T2 <= self.state_round < T2 + (self.M-1) * self.commArmLength * self.Ep_number:
            k0 = (self.state_round - T2) // (self.commArmLength * self.Ep_number) + 1  # player to com
            if self.relative_position == (k0+1):
                t0 = int(self.state_round - T2 - (k0-1) * self.commArmLength * self.Ep_number)
                index1 = t0 // self.commArmLength
                index2 = t0 % self.commArmLength
                if col:
                    self.new_arm_to_explore[index1] += 2**index2

        T3 = T2 + (self.M-1) * self.commArmLength * self.Ep_number

        self.state_round = self.state_round + 1

        if self.state_round == T3:
            self._add_log_by_round('Follower end communication') # used to track synchronization
            if self.Ep_number == 1:
                self.phase = self.EXPLOIT
                self.exploited_arm = self.new_arm_to_explore[0]
                return

            self.phase = self.EXPLORATION
            self.state_round = 0

            self.Ep_number = 0
            self.p = self.p + 1
            self.arm_to_explore = self.new_arm_to_explore

            self.new_leader_arm = 0
            self.new_communication_arm = 0










