#!/usr/bin/env python
import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv, PartialObsMAE
from multiagent.execution.q_learning import q_learning_execution
from multiagent.execution.random_policy import random_policy_execution
from multiagent.policy.td3 import run
import matplotlib.pyplot as plt


os.makedirs("./results/plots/", exist_ok=True)

def run_td3(env):
    model_path = './td3/'
    hidden_dim = 256
    rendermode = False
    episodes = 200
    steps_per_episode = 100
    warmup = 5000
    batchsize = 512
    train_rewards = run(env,
                        model_path,
                        hidden_dim,
                        batch=batchsize,
                        train=True,
                        warmup=warmup,
                        render=rendermode,
                        episodes=episodes,
                        steps_per_episode=steps_per_episode)
    test_rewards = run(env,
                       model_path,
                       hidden_dim,
                       train=False,
                       batch=batchsize,
                       warmup=warmup,
                       render=rendermode,
                       episodes=episodes,
                       steps_per_episode=steps_per_episode)

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(range(len(test_rewards)), train_rewards, label="Average Reward")  # Plot some data on the axes.
    ax.set_xlabel('Episodes')  # Add an x-label to the axes.
    ax.set_ylabel('Reward')  # Add a y-label to the axes.
    ax.set_title("Cooperative Agents")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.savefig("./plots/td3/train_rewards_collab.png")

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(range(len(test_rewards)), test_rewards, label='Average Reward')  # Plot some data on the axes.
    ax.set_xlabel('Episodes')  # Add an x-label to the axes.
    ax.set_ylabel('Average Episodic Reward')  # Add a y-label to the axes.
    ax.set_title("Cooperative Agents")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    plt.savefig("./plots/td3/test_rewards_collab.png")


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    parser.add_argument('-a', '--algorithm', default='td3', help='Name of the algorithm to run. One of q_learning, td3, mcts')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment

    # fully observable env
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)

    if args.algorithm == 'td3':
        print("Running TD3")
        run_td3(env)
    
