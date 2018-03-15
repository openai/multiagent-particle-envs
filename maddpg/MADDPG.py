from OrnsteinUhlenbeckActionNoise import OrnsteinUhlenbeckActionNoise as ou
from models import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pdb



def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self,
                 n_agents,
                 dim_obs_list,
                 dim_act_list,
                 batch_size,
                 capacity,
                 episodes_before_train,
                 load_models=None,
                 isOU=False):
        dim_obs_sum = sum(dim_obs_list)
        dim_act_sum = sum(dim_act_list)

        if load_models is None:
            self.actors = [Actor(dim_obs, dim_act) for (dim_obs, dim_act) in zip(dim_obs_list, dim_act_list)]
            self.critics = [Critic(dim_obs_sum, dim_act_sum) for i in range(n_agents)]
            self.actors_target = deepcopy(self.actors)
            self.critics_target = deepcopy(self.critics)
            self.critic_optimizer = [Adam(x.parameters(), lr=0.0075) for x in self.critics]     # 0.005
            self.actor_optimizer = [Adam(x.parameters(), lr=0.0075) for x in self.actors]       # 0.005
            self.memory = ReplayMemory(capacity)
            self.var = [1.0 for i in range(n_agents)]
            self.ou_noises = [ou(mu=np.zeros(dim_act_list[i])) for i in range(n_agents)]
        else:
            print('Start loading models!')
            states = th.load(load_models)
            self.critics = states['critics']
            self.actors = states['actors']
            self.critic_optimizer = states['critic_optimizer']
            self.actor_optimizer = states['actor_optimizer']
            self.critics_target = states['critics_target']
            self.actors_target = states['actors_target']
            self.memory = states['memory']
            self.var = states['var']
            self.ou_noises = [ou(mu=np.zeros(dim_act_list[i]), x0=states['ou_prevs'][i]) for i in range(n_agents)]
            print('Models loaded!')

        self.n_agents = n_agents
        self.batch_size = batch_size
        self.dim_obs_list = dim_obs_list
        self.dim_act_list = dim_act_list
        self.dim_obs_sum = dim_obs_sum
        self.dim_act_sum = dim_act_sum
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train
        self.clip = 50.0    # 10
        self.isOU = isOU

        self.GAMMA = 0.95
        self.tau = 0.01
        self.scale_reward = 0.01

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
        self.episode_done = 0

    def update_policy(self):
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []

        critics_grad = []
        actors_grad = []

        index_obs = 0
        index_act = 0
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            # print('transition', transitions)
            batch = Experience(*zip(*transitions))
            # print('batch.states', batch.states)
            # print('state.actions', batch.actions)
            # print('state.next_states', batch.next_states)
            non_final_mask = ByteTensor(list(map(lambda s: s is not None, batch.next_states)))
            # print('shape of non_final_mask', non_final_mask.size())
            # print('non_final_mask', non_final_mask)
            # state_batch: batch_size x dim_obs_sum
            state_batch = Variable(th.stack(batch.states).type(FloatTensor))
            # print('shape of state_batch', state_batch.size())
            # print('state_batch', state_batch)
            action_batch = Variable(th.stack(batch.actions).type(FloatTensor))
            # print('shape of action_batch', action_batch.size())
            # print('state_actions', action_batch)
            reward_batch = Variable(th.stack(batch.rewards).type(FloatTensor))
            # print('shape of reward_batch', reward_batch.size())
            # print('state_rewards', reward_batch)
            # batch_size_non_final x dim_obs_sum
            # non_final_next_states = Variable(th.from_numpy(np.array([s for s in batch.next_states if s is not None])))
            non_final_next_states = \
                Variable(th.stack([s for s in batch.next_states if s is not None]).type(FloatTensor))
            # print('non_final_next_states', non_final_next_states)

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            # print('shape of whole_state', whole_state.size())
            whole_action = action_batch.view(self.batch_size, -1)

            # critic network
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = []
            idx = 0
            for i in range(self.n_agents):
                at = self.actors_target[i](non_final_next_states[:, idx:(idx+self.dim_obs_list[i])])
                non_final_next_actions.append(at)
                idx += self.dim_obs_list[i]
            non_final_next_actions = th.cat((non_final_next_actions[0], non_final_next_actions[1]), 1)

            target_Q = Variable(th.zeros(self.batch_size).type(FloatTensor))
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.dim_obs_sum),
                non_final_next_actions.view(-1, self.dim_act_sum)
            )

            # here target_Q is y_i of TD error equation
            # target_Q = (target_Q * self.GAMMA) + (reward_batch[:, agent] * self.scale_reward)
            target_Q = target_Q * self.GAMMA + reward_batch[:, agent]

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()

            if self.clip is not None:
                nn.utils.clip_grad_norm(self.critics[agent].parameters(), self.clip)
            self.critic_optimizer[agent].step()

            # actor networkself.use_cuda
            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, index_obs:(index_obs+self.dim_obs_list[agent])]
            index_obs += self.dim_obs_list[agent]
            # print('index_obs', index_obs)
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            # print('action_i', action_i)
            # print('ac', ac)
            ac[:, index_act:(index_act+self.dim_act_list[agent])] = action_i
            index_act += self.dim_act_list[agent]
            # print('index_act', index_act)
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()

            if self.clip is not None:
                nn.utils.clip_grad_norm(self.actors[agent].parameters(), self.clip)
            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

            critics_agent_grad = []
            actors_agent_grad = []
            for x in self.critics[agent].parameters():
                critics_agent_grad.append(x.grad.data.norm(2))
                # critics_agent_grad.append(th.mean(x.grad).data[0])
            for x in self.actors[agent].parameters():
                actors_agent_grad.append(x.grad.data.norm(2))
                # actors_agent_grad.append(th.mean(x.grad).data[0])

            critics_grad.append(critics_agent_grad)
            actors_grad.append(actors_agent_grad)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return critics_grad, actors_grad

    def select_action(self, state_batch):   # no batch here!! Just concatenation of observations from agents
        # state_batch: batch_size x dim_state_sum
        actions = Variable(th.zeros(1, self.dim_act_sum))
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        index_obs = 0
        index_act = 0
        for i in range(self.n_agents):
            sb = state_batch[index_obs:(index_obs+self.dim_obs_list[i])]
            act = self.actors[i](sb)
            # print(act)


            # ? to add exploration rate here ?
            if self.isOU:   # TODO
                act += Variable(th.from_numpy(self.ou_noises[i]() * self.var[i]).type(FloatTensor))
            # else:
                # act += Variable(th.from_numpy(np.random.randn(self.dim_act_list[i]) * self.var[i]).type(FloatTensor))
                # print('act', act)

            # use more exploration??
            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999998

            act = th.clamp(act, -1.0, 1.0)
            actions[:, index_act:(index_act+self.dim_act_list[i])] = act
            # print('actions', actions)

            index_obs += self.dim_obs_list[i]
            index_act += self.dim_act_list[i]

        self.steps_done += 1

        return actions













