import numpy as np


def linear_to_goal(agent_id, env):
    agent_pos = [agent.position for agent in env.agents if agent.id == agent_id][0]
    goal_pos = [ld.position for ld in env.landmarks if ld.type == 'goal'][0]
    
    dist2 = np.sum(np.square(agent_pos - goal_pos))
    return -dist2