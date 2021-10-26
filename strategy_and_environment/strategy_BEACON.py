import numpy as np


class BEACON(object):
    """ Implementation of BEACON.
    This class is responsible for the initialization and exploration. We implement communication in banditsBEACON.py
    for simplicity. The initialization is from "Optimal algorithms for multiplayer multi-armed bandits", AISTATS2020,
    but is modified so it can handle M=K case.
    """
    def __init__(self, narms, T=100, verbose=False):
        self.name = 'BEACON'
        self.T = T
        self.t = 0
        self.K = narms
        self.M = 0  # (estimated) number of players
        self.M = 1  # (estimated) number of players
        self.relative_position = 0

        # phase
        # All players start with INIT_ORTHOG_SAMPLE. Then, they alternate between EXPLORATION and COMMUNICATION
        self.INIT_ORTHOG_SAMPLE = 1
        self.INIT_ORTHOG_VERIFICATION = 2
        self.INIT_RANK_ASSIGN = 3
        self.INITIALIZATION = 4
        self.EXPLORATION = 5
        self.EXPLORATION_SIGNAL = 6
        self.COMMUNICATION = 7

        # variables
        self.state_round = 1  # this variable is used for tracking time within one phase
        self.phase = self.INIT_ORTHOG_SAMPLE
        self.verbose = verbose

        # variables used for initialization
        self.id = 0  # 0 until fixed on some arm
        self.i = 0
        self.collisions = 0
        self.initial_explore_length = 0  # set appropriately after estimating M

        # variables for exploration
        self.cumulative_rewards = np.zeros(self.K)
        self.npulls = np.zeros(self.K, dtype=np.int32)
        self.counters = np.zeros(self.K, dtype=np.int32)
        self.old_counters = np.zeros(self.K, dtype=np.int32)
        self.empirical_means = np.zeros(self.K)
        self.last_empirical_means = np.zeros(self.K)
        self.arm_to_explore = None
        self.new_arm_to_explore = None

        # variables used for communication
        self.flag_first_communication = True
        self.commArmLength = int(np.ceil(np.log2(self.K)))  # use commArmLength bits to send x \in {0,1,2,...,K-1}
        self.communication_arm = None
        self.leader_arm = None

        # state variable
        self.is_leader = False

        # variables for leader
        self.record_counters = None
        self.record_UCB = None
        self.record_player_stat = None
        self.record_last_phase_player_stat = None
        self.record_player_npulls = None
        self.record_arm_to_explore = None

    def _update_arm_stats(self, arm, reward, col):
        self.cumulative_rewards[arm-1] += reward * (1 - col)
        self.npulls[arm-1] += 1
        # We only update the empirical_means[arm-1] when it is pulled for 2**X times.
        if self.npulls[arm-1] % 2 == 0:
            self.empirical_means[arm-1] = self.cumulative_rewards[arm-1] / self.npulls[arm-1]

    def play(self):
        a = -1
        # In sampling we choose an arm uniformly if the ID is 0, otherwise we choose id.
        if self.phase == self.INIT_ORTHOG_SAMPLE:
            a = self.id if self.id > 0 else np.random.choice(  # select one arm randomly from {1,2,...,K}
                [i + 1 for i in range(self.K)])
        # In verification, depending on the round and our ID, we choose our
        # ID or arm K.
        elif self.phase == self.INIT_ORTHOG_VERIFICATION: # K rounds
            if self.id > 0:
                a = self.id
            else:
                a = self.state_round

        # The goal of this phase is to understand how many Players
        # there are, our relative position, and if we are the leader.
        elif self.phase == self.INIT_RANK_ASSIGN: # K-1 blocks
            if self.state_round <= 2 * (self.id):
                a = self.id + 1
            else:
                a = (self.last_action + 1) % self.K + 1

        # For the first exploration phase, player explore each arm uniformly.
        elif self.phase == self.INITIALIZATION:
            a = (self.relative_position + self.state_round) % self.K + 1
        # For other exploration phases, player only explores the assigned arm.
        elif self.phase == self.EXPLORATION:
            a = self.arm_to_explore + 1

        return a - 1

    def update(self, play, obs):
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
                    self.id = self.id - 1
                    self.last_action = self.id
                    #print("My id: ", self.id)
                else: # restart sample phase
                    self.phase = self.INIT_ORTHOG_SAMPLE
                    self.state_round = 1
                    self.collisions = 0
                    #self.id = self.id - 1
            else:
                self.state_round += 1
        elif self.phase == self.INIT_RANK_ASSIGN:
            # Increase number of collisions, this will be equal to M at the end of
            # the round.
            self.last_action = play
            if col:
                self.M += 1
                if self.state_round <= 2*(self.id):
                    self.relative_position += 1

            # Finished all the rounds, move to next phase
            if self.state_round == 2 * self.K:
                # At the end of our block we check how many collisions we had. That number is the number of players
                #self.M = self.collisions + 1
                self.relative_position += 1
                if self.relative_position == 1:
                    self.is_leader = True
                self.communication_arm = self.relative_position - 1
                self.leader_arm = 0
                self.initial_explore_length = 300
                if self.is_leader: # leader has to store many things
                    self.record_player_stat = np.zeros((self.M, self.K))
                    self.record_last_phase_player_stat = np.zeros((self.M, self.K))
                    self.record_player_npulls = np.zeros((self.M, self.K))
                    self.record_matching_to_explore = np.zeros(self.M, dtype=np.int32)
                    self.record_counters = np.zeros((self.M, self.K))
                    self.record_old_counters = np.zeros((self.M, self.K))
                    self.need_communication = None
                    self.record_communication_arm = np.zeros(self.M, dtype=int)
                    for i in range(self.M):
                        self.record_communication_arm[i] = i
                    self.record_UCB = np.zeros((self.M, self.K))
                    self.explore_length = 0
                # i and state_round are used as a clock within one phase
                self.state_round = 0
                self.phase = self.INITIALIZATION
                #print("My rank: {}, My estimated M: {}".format(self.relative_position, self.M))
                #print(self.id, self.action)
                return
            self.state_round += 1

        elif self.phase == self.INITIALIZATION:
            self._update_arm_stats(arm, rewards, col)
            self.state_round += 1

            if self.state_round == self.K * self.initial_explore_length:  # end of INITIALIZATION
                if self.is_leader:
                    self.need_communication = []
                    self.phase = self.COMMUNICATION
                    for i in range(self.M):
                        if i > 0:
                            self.need_communication.append(i)
                        for j in range(self.K):
                            self.record_player_npulls[i][j] += self.initial_explore_length
                            self.record_counters[i][j] = np.floor(np.log2(self.record_player_npulls[i][j]))
                            self.record_old_counters[i][j] = 0
                else:
                    self.phase = self.COMMUNICATION
                    self.counters = np.floor(np.log2(self.npulls))

        elif self.phase == self.EXPLORATION:
            self._update_arm_stats(arm, rewards, col)
            self.state_round += 1
            if self.is_leader and self.state_round == self.explore_length:
                self.phase = self.COMMUNICATION
                self.need_communication = []
                for i in range(self.M):
                    self.record_player_npulls[i][self.record_matching_to_explore[i]] += self.explore_length
                    tmp = np.floor(np.log2(self.record_player_npulls[i][self.record_matching_to_explore[i]]))
                    if tmp > self.record_counters[i][self.record_matching_to_explore[i]]:
                        self.record_counters[i][self.record_matching_to_explore[i]] = tmp
                        if i > 0:
                            self.need_communication.append(i)



