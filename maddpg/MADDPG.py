from models import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


class MADDPG:
    def __init__(self,
                 n_agents,
                 dim_obs_list,
                 dim_act_list,
                 batch_size,
                 capacity,
                 episodes_before_train,
                 load_models=None):
        dim_obs_sum = sum(dim_obs_list)
        dim_act_sum = sum(dim_act_list)

        if load_models is None:
            self.actors = [Actor(dim_obs, dim_act) for (dim_obs, dim_act) in zip(dim_obs_list, dim_act_list)]
            self.critics = [Critic(dim_obs_sum, dim_act_sum) for i in range(n_agents)]
            self.actors_target = deepcopy(self.actors)
            self.critics_target = deepcopy(self.critics)
            self.critic_optimizer = [Adam(x.parameters(), lr=0.0001) for x in self.critics]
            self.actor_optimizer = [Adam(x.parameters(), lr=0.00001) for x in self.actors]
            self.memory = ReplayMemory(capacity)
            self.var = [1.0 for i in range(n_agents)]
        else:
            print('Start loading models!')
            states = th.load(load_models)
            self.actors = states['actors']
            self.critics = states['critics']
            self.critic_optimizer = states['critic_optimizer']
            self.actor_optimizer = states['actor_optimizer']
            self.critics_target = states['critics_target']
            self.actors_target = states['actors_target']
            self.memory = states['memory']
            self.var = states['var']
            print('Models loaded!')

        self.n_agents = n_agents
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episodes_done = 0











