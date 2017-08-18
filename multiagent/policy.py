import numpy as np
from pyglet.window import key

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyborad events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]
        # register keyboard events with this envornment's window
        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_space:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.array([0.0,0.0])
            if self.move[0]: u[0] -= 1.0
            if self.move[1]: u[0] += 1.0
            if self.move[2]: u[1] += 1.0
            if self.move[3]: u[1] -= 1.0
        c = 0
        for i in range(len(self.comm)):
            if self.comm[i]: c = i+1
        return [u, c]

    # keyborad event callbacks 
    def key_press(self, k, mod):
        if k==key.LEFT:  self.move[0] = True
        if k==key.RIGHT: self.move[1] = True
        if k==key.UP:    self.move[2] = True
        if k==key.DOWN:  self.move[3] = True
        for i in range(len(self.comm)):
            if k==key._1+i:  self.comm[i] = True
    def key_release(self, k, mod):
        if k==key.LEFT:  self.move[0] = False
        if k==key.RIGHT: self.move[1] = False
        if k==key.UP:    self.move[2] = False
        if k==key.DOWN:  self.move[3] = False
        for i in range(len(self.comm)):
            if k==key._1+i:  self.comm[i] = False