#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.envRace import MultiAgentRaceEnv
from multiagent.policy import InteractivePolicy
from multiagent.policyRace import InteractivePolicy_race
import multiagent.scenarios as scenarios

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
    env = MultiAgentRaceEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy_race(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    #single cheat)
            #print(thisAgent.action.u[1])
    while True:
    #for i in range(100):
        # query for action from each agent's policy
        
        #for i, thisAgent in enumerate(world.agents):
            #if(i==0):
                #thisAgent.action.u[1]=1
                #print(thisAgent.action.u[1])               
        act_n = [] 
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
            #print(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        #for i, thisAgent in enumerate(world.agents):
            #if(i==0):
                #print(thisAgent.action.u)        
        # render all agent views
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
