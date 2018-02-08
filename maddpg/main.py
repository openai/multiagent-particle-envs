from torch.autograd import Variable
from make_env import make_env
from MADDPG import MADDPG
import numpy as np
import torch as th
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import time


env = make_env('simple_speaker_listener')
n_agents = len(env.world.agents)
dim_obs_list = [env.observation_space[i].shape[0] for i in range(n_agents)]
dim_act_list = [env.action_space[i].n for i in range(n_agents)]

capacity = 1000000
batch_size = 1000

n_episode = 1   # 20000
max_steps = 2   # 1000
episodes_before_train = 100


maddpg = MADDPG(n_agents, dim_obs_list, dim_act_list, batch_size, capacity, episodes_before_train, load_models=None)
print('number of actors: ', len(maddpg.actors))
print('number of critics: ', len(maddpg.critics))
print('number of actors target: ', len(maddpg.actors_target))
print('number of critics target: ', len(maddpg.critics_target))
print('exploration rate: ', maddpg.var)

for i_episode in range(n_episode):
    obs = env.reset()
    for t in range(max_steps):
        env.render()
        agent_actions = []
        for i, agent in enumerate(env.world.agents):
            agent_action_space = env.action_space[i]
            action = agent_action_space.sample()
            action_vec = np.zeros(agent_action_space.n)
            action_vec[action] = 1
            agent_actions.append(action_vec)

        time.sleep(0.033)
        observation, reward, done, info = env.step(agent_actions)

        print(agent_actions)
        print(observation)
        print(reward)
        print(done)
        print(info)
        print()









