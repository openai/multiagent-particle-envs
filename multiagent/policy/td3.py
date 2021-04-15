# Won't work yet. Pls don't use
import torch
import random
import copy
import time
import os
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F
from torch.utils import tensorboard
import numpy as np

# td3writer = tensorboard.SummaryWriter()

class ReplayBuffer(object):
    def __init__(self):
        self.buffer = []

    def add(self, obslist):
        self.buffer.append(obslist)

    def sample(self, sample_size=1):
        return random.sample(self.buffer, sample_size)

    def reset(self):
        self.buffer = []

class QNetwork(nn.Module):
    # Takes state and action as input and outputs the expected Q value
    def __init__(self, statedim, actiondim, hiddendim):
        super().__init__()

        self.linear1 = nn.Linear(statedim+actiondim, hiddendim)
        self.linear2 = nn.Linear(hiddendim, hiddendim)
        self.linear3 = nn.Linear(hiddendim, 1)

    def forward(self, state, action):
        # print(f"{state=}, {action=}")
        stateact = torch.cat([state, action], dim=-1)
        x = F.relu(self.linear1(stateact))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNet(nn.Module):
    def __init__(self, statedim, hiddendim, actiondim):
        super().__init__()
        # print(f"{actiondim=}")
        self.linear1 = nn.Linear(statedim, hiddendim)
        self.linear2 = nn.Linear(hiddendim, hiddendim)
        self.linear3 = nn.Linear(hiddendim, actiondim)

    def forward(self, state):
        # Forward evaluation
        # Returns actiondim*2 values - 
        # each dimension forward backward movement 
        # clamped to (-1,1)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x)) 
        # print(x)
        return x

    def clip(self, val, cliplim):
        return torch.clamp(val, -cliplim, cliplim)

    def onehot_from_logits(self, logits):
        argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
        return argmax_acs

    def gumbel_softmax_sample(self, logits, temperature):
        y= logits + distributions.Gumbel(0, 1).sample(sample_shape=logits.shape)
        return F.softmax(y/temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = self.onehot_from_logits(y)
            y = (y_hard - y).detach() + y
        return y

    def select_action(self, state, clip=0.5):
        # Action changing the env
        noise_dist = distributions.Normal(0, 1)
        x = self.forward(torch.Tensor(state))

        action = x.detach().cpu()
        noise = noise_dist.sample(action.shape)
        noise = self.clip(noise, cliplim=clip)

        action = action+noise
        #Convert to categorical
        action = self.gumbel_softmax(action, 1)
        return action
    
    def eval_action(self, state, clip=0.5):
        # Action to be evaluated when sampling from the replay buffer
        normal = distributions.Normal(0, 1) #noise distribution

        x = self.forward(state)
        noise = self.clip(normal.sample(x.shape), cliplim=clip)
        action = x+noise

        action = self.gumbel_softmax(action, 1)
        
        return action, noise, x

class TD3(object):
    def __init__(self, statedim, actiondim, hiddendim, name='Agent', target_update_ctr=10):
        super().__init__()
        
        self.name = name

        #Counters to track training loops
        self.update_ctr = 0
        self.target_update_ctr = target_update_ctr
        actiondim = 2*actiondim

        self.critic_1 = QNetwork(statedim, actiondim, hiddendim)
        self.critic_2 = QNetwork(statedim, actiondim, hiddendim)
        self.actor = PolicyNet(statedim, hiddendim, actiondim)

        #Initialize targets to same parameters as corresponding nets
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        self.target_actor = copy.deepcopy(self.actor)

        # Replay buffer
        self.replaybuf = ReplayBuffer()

        #Optimizers
        self.optimizer_critic_1 = optim.Adam(self.critic_1.parameters(), lr=1e-4)
        self.optimizer_critic_2 = optim.Adam(self.critic_2.parameters(), lr=1e-4)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=1e-4)

    def episode_reset(self):
        self.replaybuf.reset()

    def target_network_update(self, target, net, tau=0.005):

        for tgtparam, netparam in zip(target.parameters(), net.parameters()):
            tgtparam.data.copy_(tau * netparam.data + (1 - tau) * tgtparam.data)

        return target

    def update(self, batch, gamma=0.99):

        sarsa = self.replaybuf.sample(batch)

        state = torch.Tensor(np.array(sarsa[0][0]))
        action = torch.Tensor(np.array(sarsa[0][1]).reshape(-1))
        reward = torch.Tensor(np.array(sarsa[0][2]))
        next_state = torch.Tensor(np.array(sarsa[0][3]))
        terminal = torch.Tensor(np.array(sarsa[0][4]))

        next_action, _, _ = self.actor.eval_action(next_state)
        next_tgt_action, _, _ = self.target_actor.eval_action(next_state)

        pred_q_tgt = torch.min(self.target_critic_1(next_state, next_tgt_action),
                               self.target_critic_2(next_state, next_tgt_action))

        y = reward + (1-terminal) * gamma * pred_q_tgt
        c1pred = self.critic_1(state, action)
        c2pred = self.critic_2(state, action)
        # td3writer.add_scalar("critic1",c1pred)
        # loss_q1 = ((y.detach() - c1pred)**2).mean()
        # loss_q2 = ((y.detach() - c2pred)**2).mean()
        loss_q1 = F.smooth_l1_loss(y.detach(), c1pred)
        loss_q2 = F.smooth_l1_loss(y.detach(), c2pred)

        #Update critic 1
        self.optimizer_critic_1.zero_grad()
        loss_q1.backward()
        self.optimizer_critic_1.step()

        #Update critic 2
        self.optimizer_critic_2.zero_grad()
        loss_q2.backward()
        self.optimizer_critic_2.step()


        if self.update_ctr%self.target_update_ctr==0:
            #Update the actor network
            pred_q = self.critic_1(state, next_action)

            # policy_loss = -pred_q.mean()
            policy_loss = pred_q.mean()
            # print(policy_loss)
            self.optimizer_actor.zero_grad()
            policy_loss.backward()
            self.optimizer_actor.step()
            # Update the target networks using tau
            # network_param = tau * network_param + (1-tau) * network_param
            self.target_network_update(self.target_critic_1, self.critic_1)
            self.target_network_update(self.target_critic_2, self.critic_2)
            self.target_network_update(self.target_actor, self.actor)

            # print(f"{reward=},\n{y=},\n{pred_q_tgt=},\n{c1pred=},\n{c2pred=},\n{loss_q1=},\n{loss_q2=},\n{policy_loss=}")        
        self.update_ctr += 1
    
    def model_files(self, disk_path):

        critic_1_fname = os.path.join(disk_path,"_".join([self.name,"critic_1"]))
        critic_2_fname = os.path.join(disk_path,"_".join([self.name,"critic_2"]))
        actor_fname = os.path.join(disk_path,"_".join([self.name,"actor"]))

        return critic_1_fname, critic_2_fname, actor_fname

    def save_model(self, save_path):
        
        os.makedirs(os.path.join(save_path), exist_ok=True)

        critic_1_fname, critic_2_fname, actor_fname = self.model_files(save_path)

        torch.save(self.critic_1.state_dict(), critic_1_fname)
        torch.save(self.critic_2.state_dict(), critic_2_fname)
        torch.save(self.actor.state_dict(), actor_fname)
    
    def load_model(self, load_path):
        
        critic_1_fname, critic_2_fname, actor_fname = self.model_files(load_path)

        self.critic_1.load_state_dict(torch.load(critic_1_fname))
        self.critic_2.load_state_dict(torch.load(critic_2_fname))
        self.actor.load_state_dict(torch.load(actor_fname))

        self.critic_1.eval()
        self.critic_2.eval()
        self.actor.eval()

def run(env,
        model_path,
        hiddendim=128,
        steps_per_episode=1000,
        episodes=100,
        batch=5000,
        model_prefix="Agent",
        train=True,
        render=False):

    print("Running TD3 in training mode")
    networks = []
    batchctr = batch
    updates = 0
    if train:
        for i in range(env.n):
            networks.append(TD3(env.observation_space[i].shape[0],
                                2,#env.action_space[i].shape[0],
                                hiddendim,
                                f"{model_prefix}{i}",
                                target_update_ctr=2))
        
        total_rewards = []
        bufctr = 0
        for episode in range(episodes):
            # bufctr = 0
            states = env.reset()
            start = time.time()
            ep_rewards = np.array([0.]*env.n)

            for step in range(steps_per_episode):
                # print(f"{episode=},{step=}")  
                actions = []
                envactions = []
                for i, network in enumerate(networks):
                    # print(f"{actions=}")
                    action = network.actor.select_action(states[i])
                    envaction = int(action.argmax())+1 # Returning only the integer action value
                                                       # Discrete inputs only
                    envactions.append(envaction)
                    actions.append(action)
                
                if render: env.render()
                # print(envactions)
                next_states, reward_n, done_n, info_n = env.step(envactions)

                if step == steps_per_episode-1:
                    done_n = np.array([1.]*env.n)
                else:
                    done_n = np.array([0.]*env.n)

                states = next_states
                ep_rewards += reward_n
                # print(actions, reward_n)
                
                # Replay buffer update
                for i, network in enumerate(networks):
                    network.replaybuf.add([states[i], actions[i], reward_n[i], next_states[i], done_n[i]])

                bufctr+=1

                if bufctr > batchctr:
                    updates+=1
                    for i, network in enumerate(networks):
                        #Update the parameters after 
                        network.update(batch)

                
            total_rewards.append(ep_rewards)
            # td3writer.add_scalar("Reward/EP",ep_rewards, episode)
            eptime = time.time() - start
            print(f"Episode {episode} finished with rewards {ep_rewards}. Time taken: {eptime}")
        print(f"Total target updates:{updates}")
        
        for network in networks:
            network.save_model(model_path)
        # td3writer.flush()

    else:
        print("Running TD3 in eval mode")
        networks = []
        for i in range(env.n): # Load from disk
            net = TD3(env.observation_space[i].shape[0],
                      2,#env.action_space[i].shape[0],
                      hiddendim,
                      f"{model_prefix}{i}")
            net.load_model(model_path)
            networks.append(net)

        states = env.reset()
        total_rewards = []
        for episode in range(episodes):
            start = time.time()
            ep_rewards = np.array([0.]*env.n)
            for step in range(steps_per_episode):
                actions = []
                envactions = []
                for i, network in enumerate(networks):
                    action = network.actor.select_action(states[i])
                    envaction = int(action.argmax())+1 # Returning only the integer action value
                                                    # Discrete inputs only
                    envactions.append(envaction)
                    actions.append(action)

                if render: env.render()
                next_states, reward_n, done_n, info_n = env.step(envactions)

                states = next_states
                ep_rewards += reward_n
            eptime = time.time() - start
            print(f"Episode {episode} finished with rewards {ep_rewards}. Time taken: {eptime}")
            total_rewards.append(ep_rewards)
        
    return total_rewards