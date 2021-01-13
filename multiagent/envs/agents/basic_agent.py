import multiagent.envs.observations as observations
import multiagent.envs.rewards as rewards
import multiagent.envs.physics as physics
import numpy as np

from gym.spaces import Box

class BasicAgent():
    def __init__(self, id="agent", initial_conditions=None, physic="BasicPhysic",
                 observation="simple_goal", reward="linear_to_goal"):
        
        self.id = id
        # Data used to reset agent
        self._init_pos = initial_conditions['position']\
            if initial_conditions is not None and 'position' in initial_conditions\
                else np.zeros(2)  # [x, y]
        self._init_velocity = initial_conditions['velocity']\
            if initial_conditions is not None and 'velocity' in initial_conditions\
                else np.zeros(2)  # [x, y]
        self._init_color = initial_conditions['color']\
            if initial_conditions is not None and 'color' in initial_conditions\
                else np.zeros(4)  # [r, g , b, alpha]
        self._init_movable = initial_conditions['movable']\
            if initial_conditions is not None  and 'movable' in initial_conditions\
                else True
        self._init_max_speed = initial_conditions['max_speed']\
            if initial_conditions is not None  and 'max_speed' in initial_conditions\
                else None
        self._init_size = initial_conditions['size']\
            if initial_conditions is not None  and 'size' in initial_conditions\
                else 0.02
        
        # Agent functions and classes
        self._observation = getattr(observations, observation)
        self._reward = getattr(rewards, reward)
        self._physic = getattr(physics, physic)()
        
        self.actions = np.zeros(2)  # [x, y], de -1 à 1
        
        self.action_space = Box(low=-1, high=+1, shape=self.actions.shape, dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32)
        
        # Init values by reseting agent
        self.reset()

    def reset(self):
        self.position = self._init_pos
        self.velocity = self._init_velocity
        self.color = self._init_color
        self.movable = self._init_movable
        self.max_speed = self._init_max_speed
        self.size = self._init_size
        
    def observation(self, env):
        return self._observation(self.id, env)
        
    def reward(self, env):
        return self._reward(self.id, env)
    
    def step(self, action, env):
        self.actions = action
        
        self._physic.move(self.id, self.actions, env)
