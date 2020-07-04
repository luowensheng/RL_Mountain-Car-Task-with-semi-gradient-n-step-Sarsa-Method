import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import floor
import pdb

#######################################################################
# Following are some utilities for tile coding from Rich.             #
# To make each file self-contained, I copied them from                #
# http://incompleteideas.net/tiles/tiles3.py-remove                   #
# with some naming convention changes                                 #
#                                                                     #
# Please complete the following parts:                                #
#         1. "def learn"                                              #
#         2. "def semi_gradient_n_step_sarsa"                         #
#######################################################################

# Tile coding starts  (You can't edit this part) ######################
class IHT:
    "Structure to handle collisions"
    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count

def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT): return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int): return hash(tuple(coordinates)) % m
    if m is None: return coordinates

def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles
# Tile coding ends ####################################################

# all possible actions
ACTION_REVERSE = -1
ACTION_ZERO = 0
ACTION_FORWARD = 1
# order is important
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]

# bound for position and velocity
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
EPSILON = 0

# take an action at position and velocity
# return: new position, new velocity, reward (always -1)
def step(position, velocity, action):
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    new_velocity = min(max(VELOCITY_MIN, new_velocity), VELOCITY_MAX)
    new_position = position + new_velocity
    new_position = min(max(POSITION_MIN, new_position), POSITION_MAX)
    reward = -1.0
    if new_position == POSITION_MIN:
        new_velocity = 0.0
    return new_position, new_velocity, reward

# wrapper class for state action value function
class ValueFunction:
    # max_size: the maximum # of indices
    def __init__(self, step_size, num_of_tilings=8, max_size=2048):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings

        # divide step size equally to each tiling
        self.step_size = step_size / num_of_tilings

        self.hash_table = IHT(max_size)

        # weight for each tile
        self.weights = np.zeros(max_size)

        # position and velocity needs scaling to satisfy the tile software
        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    # get indices of active tiles for given state and action
    def get_active_tiles(self, position, velocity, action):
        # I think positionScale * (position - position_min) would be a good normalization.
        # However positionScale * position_min is a constant, so it's ok to ignore it.
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                            [self.position_scale * position, self.velocity_scale * velocity],
                            [action])
        return active_tiles

    # estimate the value of given state and action
    def value(self, position, velocity, action):
        if position == POSITION_MAX:
            return 0.0
        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    # learn with given state, action and target
    def learn(self, position, velocity, action, target):
        ##########################################################################
        # TO DO:                                                                 #
        # Implement the semi-gradient n-step Sarsa update rule                   #
        # Hint:                                                                  #
        # On the text book algorithm 10.2, "self.weights[active_tiles]" is w,    #
        #                                  "target" is G,                        #
        #                                  "self.step_size" is alpha,            #
        #                    sum of "self.weights[active_tiles]" is q^(S,A,w).   #
        #                                                                        # 
        # Return: NULL                                                           #                             
        ##########################################################################
        active_tiles = self.get_active_tiles(position, velocity, action)
        for element in active_tiles:
            self.weights[element] += self.step_size * ( target - np.sum(self.weights[active_tiles]) )

        ########################################################################## 
        # Your code write here                                                   # 
        ##########################################################################
       


# get action at position and velocity based on epsilon greedy policy and valueFunction
def get_action(position, velocity, value_function):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    values = []
    for action in ACTIONS:
        values.append(value_function.value(position, velocity, action))
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)]) - 1



# semi-gradient n-step Sarsa
# valueFunction: state value function to learn
# n: # of steps
def semi_gradient_n_step_sarsa(value_function, n=1):
    # start at a random position around the bottom of the valley
    current_position = np.random.uniform(-0.6, -0.4)
    # initial velocity is 0
    current_velocity = 0.0
    # get initial action
    current_action = get_action(current_position, current_velocity, value_function)
    # track previous position, velocity, action and reward
    positions = [current_position]
    velocities = [current_velocity]
    actions = [current_action]
    rewards = [0.0]

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')
    while True:
        # go to next time step
        time += 1
        if time < T:
            # take current action and go to the new state
            new_postion, new_velocity, reward = step(current_position, current_velocity, current_action)
            # choose new action
            new_action = get_action(new_postion, new_velocity, value_function)

            # track new state and action
            positions.append(new_postion)
            velocities.append(new_velocity)
            actions.append(new_action)
            rewards.append(reward)
            #################################################################
            # TO DO:                                                        #
            # Implement the semi-gradient n-step Sarsa algorithm            #
            # You can refer to text book algorithm 10.2                     #
            # Hint: update_time = time - n                                  #
            #       if update_time >= 0 ...                                 #
            #################################################################
            if new_postion == POSITION_MAX:
                T = time
        ########################################################################## 
        # Your code write here                                                   # 
        ##########################################################################
        # Steps:                                                                 #
        #       1. Get the time of the state to update                           #
        #       2. Calculate corresponding rewards                               #
        #       3. Estimated state action value to the return                    #
        #       4. Update the state value function                               #
        ##########################################################################
        
        # 1. Get the time of the state to update
        update_time = time - n
        if update_time >= 0:
            returns = 0.0
            
           # active_tiles = value_function.get_active_tiles(positions[update_time], velocities[update_time], actions[update_time])
            # 2. calculate corresponding rewards
            ########################################################################## 
            # Your code write here                                                   # 
            ##########################################################################
            returns = np.sum([rewards[i] for i in range( update_time,min(update_time+n, T ) + 1) ])
            
            # 3. add estimated state action value
            ########################################################################## 
            # Your code write here                                                   # 
            ##########################################################################
            if update_time+n <= T:
                returns += value_function.value(positions[update_time+n], velocities[update_time+n], actions[update_time+n])
            #returns=target
            # 4. update the state value function
            ########################################################################## 
            # Your code write here                                                   # 
            # Hint:                                                                  #    
            #      if positions[update_time] != POSITION_MAX: ...                    #
            ##########################################################################
            if positions[update_time] != POSITION_MAX:
               value_function.learn(positions[update_time], velocities[update_time], actions[update_time], returns)        
            
        if update_time == T-1:    
           break 
        current_position = new_postion
        current_velocity = new_velocity
        current_action = new_action
   
        
    return time

# one-step semi-gradient Sarsa vs 8-step semi-gradient Sarsa    

def main():
    runs = 10
    episodes = 500
    num_of_tilings = 8
    alphas = [0.5, 0.3]
    n_steps = [1, 8]

    steps = np.zeros((len(alphas), episodes))
    for run in range(runs):
        value_functions = [ValueFunction(alpha, num_of_tilings) for alpha in alphas]
        for index in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                step = semi_gradient_n_step_sarsa(value_functions[index], n_steps[index])
                steps[index, episode] += step
                
    steps /= runs

    for i in range(0, len(alphas)):
        plt.plot(steps[i], label='n = %.01f' % (n_steps[i]))
    plt.xlabel('Episode')
    plt.ylabel('Steps per episode')
    plt.yscale('log')
    plt.legend()

    plt.savefig('semi_gradient_Sarsa.png')
    plt.close()


if __name__ == '__main__':
    main()