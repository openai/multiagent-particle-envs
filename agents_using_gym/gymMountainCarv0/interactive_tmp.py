#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

import numpy as np

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    EPISODE = 25000

    def get_discrete_state(state):
        #DISCRETE_OBS_SPACE = [20] * len(state)
        high_bound = np.array([1] * len(state))
        low_bound = np.array([-1] * len(state))
        obs_win_size = (high_bound-low_bound) / ([20]* len(state))
        discrete_state = np.subtract(state, low_bound)/ obs_win_size
        #print(discrete_state.astype(np.float))
        # we use this tuple to look up the 3 Q values for the available actions in the q-table
        return tuple(discrete_state.astype(np.int))

    # execution loop
    obs_n = env.reset()

    #a list of q_tables (one q_table for each agent)

    DISCRETE_OBS_SPACE = [20] * len(obs_n[0])
    q_tables = []
    for i in range(env.n):
        q_tables.append(np.random.uniform(low=-3, high=3, size=(DISCRETE_OBS_SPACE + [4])))
    q_tables = np.array(q_tables)
    #print(q_tables)

    
    #for i in range(EPISODE): do the following
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
            new_discrete_state = get_discrete_state(obs_n[i])

        print(act_n)
        #print(obs_n)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))

        
        if True:
            for i, policy in enumerate(policies):
                #print(q_tables[tuple([0])+(new_discrete_state,)])
                max_future_q = np.max(q_tables[tuple([i])+new_discrete_state])
                current_q = q_tables[tuple([i])+new_discrete_state]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward_n[i] + DISCOUNT * max_future_q)
                q_tables[tuple([i])+ new_discrete_state+(act_n[i], )] = new_q
        
    
