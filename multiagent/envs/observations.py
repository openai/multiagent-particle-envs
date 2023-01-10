import numpy as np


def self_observation(agent_id, env):
    return np.concatenate((env.agents[agent_id].position,
                           env.agents[agent_id].velocity,
                           env.agents[agent_id].color), axis=0)


def simple_goal(agent_id, env):
    agent = [agent for agent in env.agents if agent.id == agent_id][0]
    goal = [ld for ld in env.landmarks if ld.type == 'goal'][0]
    
    return np.concatenate((agent.position,
                           agent.velocity,
                           goal.position), axis=0)
    