import numpy as np


class BasicPhysic():
    
    def __init__(self):
        # physical damping
        self.damping = 0.25
        # simulation timestep
        self.dt = 0.1
    
    def move(self, agent_id, action, env):
        agent = [agent for agent in env.agents if agent.id == agent_id][0]
        if agent.movable is False:
            return
        agent.velocity = agent.velocity * (1 - self.damping)
        agent.velocity += action * self.dt
        
        if agent.max_speed is not None:
            speed = np.sqrt(np.square(agent.velocity[0]) + np.square(agent.velocity[1]))
            if speed > agent.max_speed:
                agent.velocity = agent.velocity / speed * agent.max_speed
                
        # Avoid read-only numpy array
        pos = np.copy(agent.position)
        pos += agent.velocity * self.dt
        agent.position = pos
