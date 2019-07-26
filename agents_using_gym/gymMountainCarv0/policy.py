import numpy as np
from pyglet.window import key

from multiagent.scenarios.simple import Scenario

# individual agent policy
class Policy(object):
    def __init__(self):
        self.move = [False for i in range(4)]
    def action(self, obs):
        #agent = env.agents
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

        '''
        If we try to implement Q-learning in Interactive.action(self, obs),
        we may first need to have a get_reward() function for each agent.

        Or a simpler way is to have Interactive.action(self, obs) return the action space
        each time. Then implement the Q-learning algorithm in bin/interactive.py since interactive.py have access to everything
        and it's more convinient to implement.
        '''
        
        #obs[2] is the x-axis of the relative position between first landmark and the agent
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
            if self.move[0]: u[1] += 1.0
            if self.move[1]: u[2] += 1.0
            if self.move[3]: u[3] += 1.0
            if self.move[2]: u[4] += 1.0
            if True not in self.move:
                u[0] += 1.0
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
