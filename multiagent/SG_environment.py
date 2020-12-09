import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

import numpy as np
import random

import tensorflow as tf

from gym_socialgame.envs.utils import price_signal, fourier_points_from_action
from gym_socialgame.envs.agents import *
from gym_socialgame.envs.reward import Reward

import pickle

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'] # ?? necc?
    }

    def __init__(
    	self,
        action_space_string = "continuous", 
        response_type_string = "l", 
        number_of_participants = 10,
        one_day = 0, 
        energy_in_state = False, 
        yesterday_in_state = False,
        day_of_week = False,
        pricing_type="TOU",
        reward_function = "scaled_cost_distance",
        fourier_basis_size = 10,
    	):

    # self.world = world  ## does world appear elsewhere 

    player_dict = self._create_agents()
    self.agents = player_dict # or transform however needed # world.policy_agents
    self.n = len(self.agents)

    self.discrete_action_space = False
    self.discrete_action_input = False
    self.force_discrete_action = False

    self.shared_reward = False ## right? 
    self.time = 0

	self.points_length = 10
	self.action_length = 10
	self.action_subspace = 3

	self.action_space = []
	

	# (our inputs)
	self.prev_energy = np.zeros(10)
    self.one_day = self._find_one_day(one_day)
    self.energy_in_state = energy_in_state
    self.yesterday_in_state = yesterday_in_state
    self.reward_function = reward_function
    self.fourier_basis_size = fourier_basis_size

    self.day = 0
    self.days_of_week = [0, 1, 2, 3, 4]
    self.day_of_week_flag = day_of_week
    self.day_of_week = self.days_of_week[self.day % 5]

    #Create Observation Space (aka State Space)
    self.observation_space = []

    if pricing_type=="TOU":
        self.pricing_type = "time_of_use"
    elif pricing_type == "RTP":
        self.pricing_type = "real_time_pricing"
    else:
        print("Wrong pricing type")
        raise ValueError
    
    self.prices = self._get_prices()

    ## back to filling in their skeleton 

    for agent in self.agents:
        # action seq isn't continuous, so skipping Multidiscrete action
        self.action_space.append(self._create_action_space())

        obs_dim = 10 + 10 * self.energy_in_state + 10 * self.yesterday_in_state
        self.observation_space.append(self._create_observation_space())
        
  
    def _find_one_day(self, one_day: int):
            """
            Purpose: Helper function to find one_day to train on (if applicable)

            Args:
                One_day: (Int) in range [-1,365]

            Returns:
                0 if one_day = 0
                one_day if one_day in range [1,365]
                random_number(1,365) if one_day = -1
            """
            
            print("one_day")
            print(one_day)

            # if(one_day != -1):
            #     return np.random.randint(0, high=365)
            
            # else:
            return one_day


    def _create_action_space(self):
        """
        Purpose: Return action space of type specified by self.action_space_string

        Args:
            None
        
        Returns:
            Action Space for environment based on action_space_str 
        
        Note: Multidiscrete refers to a 10-dim vector where each action {0,1,2} represents Low, Medium, High points respectively.
        We pose this option to test whether simplifying the action-space helps the agent. 
        """

        #Making a symmetric, continuous space to help learning for continuous control (suggested in StableBaselines doc.)
        if self.action_space_string == "continuous":
            return spaces.Box(low=-1, high=1, shape=(self.action_length,), dtype=np.float32)

        elif self.action_space_string == "multidiscrete":
            discrete_space = [self.action_subspace] * self.action_length
            return spaces.MultiDiscrete(discrete_space)

        elif self.action_space_string == "fourier":
            return spaces.Box(
                low=-2, high=2, shape=(2*self.fourier_basis_size - 1,), dtype=np.float32
            )


    def _create_agents(self):
        """
        Purpose: Create the participants of the social game. We create a game with n players, where n = self.number_of_participants

        Args:
            None

        Returns:
              agent_dict: Dictionary of players, each with response function based on self.response_type_string

        """

        player_dict = {}

        # Sample Energy from average energy in the office (pre-treatment) from the last experiment 
        # Reference: Lucas Spangher, et al. Engineering  vs.  ambient  typevisualizations:  Quantifying effects of different data visualizations on energy consumption. 2019
        
        sample_energy = np.array([ 0.28,  11.9,   16.34,  16.8,  17.43,  16.15,  16.23,  15.88,  15.09,  35.6, 
                                123.5,  148.7,  158.49, 149.13, 159.32, 157.62, 158.8,  156.49, 147.04,  70.76,
                                42.87,  23.13,  22.52,  16.8 ])

        #only grab working hours (8am - 5pm)
        working_hour_energy = sample_energy[8:18]

        my_baseline_energy = pd.DataFrame(data = {"net_energy_use" : working_hour_energy})

        for i in range(self.number_of_participants):
            player = DeterministicFunctionPerson(my_baseline_energy, points_multiplier = 10, response= self.response_type_string) #, )
            ## TODO: Deterministic Function needs to be modified to set a flag on each agent to silent
            ## TODO: add a prev_energy to keep track of their yesterday's energy

            player_dict['player_{}'.format(i)] = player

        return player_dict

def _create_observation_space(self):
    """
    Purpose: Returns the observation space. 
    If the state space includes yesterday, then it is +10 dim for yesterday's price signal
    If the state space includes energy_in_state, then it is +10 dim for yesterday's energy

    Args:
        None

    Returns:
        Action Space for environment based on action_space_str 
    """

    #TODO: Normalize obs_space !
    if(self.yesterday_in_state):
        if(self.energy_in_state):
            return spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
        else:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    else:
        if self.energy_in_state:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        else:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

def _get_prices(self):
        """
        Purpose: Get grid price signals for the entire year (using past data from a building in Los Angeles as reference)

        Args:
            None
            
        Returns: Array containing 365 price signals, where array[day_number] = grid_price for day_number from 8AM - 5PM

        """
        all_prices = []
        print("--" * 10)
        print(self.one_day)
        print("--" * 10)
        
        type_of_DR = self.pricing_type

        if self.one_day != -1:
            # If one_day we repeat the price signals from a fixed day
            # Tweak One_Day Price Signal HERE
            price = price_signal(self.one_day, type_of_DR=type_of_DR)
            price = np.array(price[8:18])
            if np.mean(price)==price[2]:
                price[3:6]+=.3
            price = np.maximum(0.01 * np.ones_like(price), price)

            for i in range(365):
                all_prices.append(price)
        else:
            day = 0
            for i in range(365):  
                price = price_signal(day + 1, type_of_DR=type_of_DR)
                price = np.array(price[8:18])
                # put a floor on the prices so we don't have negative prices
                if np.mean(price)==price[2]:
                    price[3:6]+=.3
                price = np.maximum(0.01 * np.ones_like(price), price)
                all_prices.append(price)
                day += 1

        return np.array(all_prices)

def _simulate_human(self, action, agent):
        """
        Purpose: Gets energy consumption from players given action from agent

        Args:
            Action: 10-dim vector corresponding to action for each hour 8AM - 5PM
        
        Returns: 
            Energy_consumption: Dictionary containing the energy usage by player and the average energy used in the office (key = "avg")
        """

        #Get players response to agent's actions
        player = agent

        if (self.day_of_week_flag):
            player_energy = player.get_response(action, day_of_week = self.day_of_week)
        else: 
            player_energy = player.get_response(action, day_of_week = None)

            #Calculate energy consumption by player and in total (over the office)
            
        return player_energy


def step(self, action_n):
    obs_n = []
    reward_n = []
    done_n = []
    info_n = {"n": []}

    for i, agent in enumerate(self.agents):
        agent.action = action_n[i]
    
    prev_price = self.prices[(self.day)]
        self.day = (self.day + 1) % 365
        self.curr_iter += 1
        
        if self.curr_iter > 0:
            done = True
            self.curr_iter = 0
        else:
            done = False

    for agent in self.agents:
            obs_n.append(self._get_observation_per_agent(agent))
            reward_n.append(self._get_reward_per_agent(agent))
            done_n.append(self._get_done(agent))

            info_n["n"].append(self._get_info(agent))

    reward = np.sum(reward_n)
    return obs_n, reward_n, done_n, info_n

def _get_observation_per_agent(self, agent):
        prev_price = self.prices[ (self.day - 1) % 365]
        next_observation = self.prices[self.day]

        if(self.yesterday_in_state):
            if self.energy_in_state:
                return np.concatenate((next_observation, np.concatenate((prev_price, self.prev_energy))))
            else:
                return np.concatenate((next_observation, prev_price))

        elif self.energy_in_state:
            return np.concatenate((next_observation, self.prev_energy))

        else:
            return next_observation

def _get_reward_per_agent(self, agent, price, energy_consumptions, reward_function = "scaled_cost_distance"):
        # TODO: finish massaging this 
        """
        Purpose: Compute reward given price signal and energy consumption of the office

        Args:
            Price: Price signal vector (10-dim)
            Energy_consumption: Dictionary containing energy usage by player in the office and the average office energy usage
        
        Returns: 
            Energy_consumption: Dictionary containing the energy usage by player and the average energy used in the office (key = "avg")
        """

        total_reward = 0
        for player_name in energy_consumptions:
            if player_name != "avg":
                # get the points output from players
                player = self.player_dict[player_name]

                # get the reward from the player's output
                player_min_demand = player.get_min_demand()
                player_max_demand = player.get_max_demand()
                player_energy = energy_consumptions[player_name]
                player_reward = Reward(player_energy, price, player_min_demand, player_max_demand)

                if reward_function == "scaled_cost_distance":
                    player_ideal_demands = player_reward.ideal_use_calculation()
                    reward = player_reward.scaled_cost_distance(player_ideal_demands)

                elif reward_function == "log_cost_regularized":
                    reward = player_reward.log_cost_regularized()

                total_reward += reward

        return total_reward
    
    def reset(self):
        # reset world
        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_observation_per_agent(agent))
        return obs_n

    def render(self, mode='human'):
        pass

    def close(self):
        pass