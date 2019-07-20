import numpy as np
from pyglet.window import key

from multiagent.scenarios.simple import Scenario

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        #self.agent_index = agent_index
        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this environment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events

        
        #x_axis = self.env.agents[self.agent_index].state.p_pos[0]
        #y_axis = self.env.agents[self.agent_index].state.p_pos[1]

        if obs[2] < 0:
            self.move[1] = True
        elif obs[2] > 0:
            self.move[0] = True
        else:
            self.move[0] = False
            self.move[1] = False

        if obs[3] > 0:
            self.move[3] = True
        elif obs[3] < 0:
            self.move[2] = True
        else:
            self.move[2] = False
            self.move[3] = False
        

        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 0.01
            if self.move[1]: u[2] += 0.01
            if self.move[3]: u[3] += 0.01
            if self.move[2]: u[4] += 0.01
            if True not in self.move:
                u[0] += 0.01
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks
    def key_press(self, k, mod):
        if k==key.LEFT:  self.move[0] = True
        if k==key.RIGHT: self.move[1] = True
        if k==key.UP:    self.move[2] = True
        if k==key.DOWN:  self.move[3] = True
    def key_release(self, k, mod):
        if k==key.LEFT:  self.move[0] = False
        if k==key.RIGHT: self.move[1] = False
        if k==key.UP:    self.move[2] = False
        if k==key.DOWN:  self.move[3] = False
