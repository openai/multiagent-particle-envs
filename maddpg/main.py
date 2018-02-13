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

capacity = 10000
batch_size = 5

n_episode = 100   # 20000
max_steps = 5   # 1000
episodes_before_train = 5

reward_record = []

maddpg = MADDPG(n_agents, dim_obs_list, dim_act_list, batch_size, capacity, episodes_before_train, load_models=None)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

for i_episode in range(n_episode):
    obs = env.reset()
    # obs = [obs[i] for i in range(n_agents)]
    obs = np.concatenate(obs, 0)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()    # obs in Tensor now
    total_reward = 0.0
    av_critics_grad = np.zeros((n_agents, 6))
    av_actors_grad = np.zeros((n_agents, 6))
    n = 0
    for t in range(max_steps):
        print(t)
        env.render()

        # obs turns to Variable before feed into Actor
        obs = Variable(obs).type(FloatTensor)
        # print('obs', obs)

        action = maddpg.select_action(obs).data.cpu()
        action = [action[0].numpy()[:dim_act_list[0]], action[0].numpy()[dim_act_list[0]:]]
        obs_, reward, done, _ = env.step(action)

        action = np.concatenate(action, 0)
        action = th.from_numpy(action).float()
        # print('action', action)

        reward = th.FloatTensor(reward).type(FloatTensor)

        # obs_ = [obs_[i] for i in range(n_agents)]
        obs_ = np.concatenate(obs_, 0)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()

        maddpg.memory.push(obs.data, action, next_obs, reward)  # tensors
        # print('obs', obs.data)
        # print('action', action)
        # print('next_obs', next_obs)
        # print('reward', reward)

        obs = next_obs

        critics_grad, actors_grad = maddpg.update_policy()

        if maddpg.episode_done > maddpg.episodes_before_train:
            av_critics_grad += np.array(critics_grad)
            av_actors_grad += np.array(actors_grad)
            n += 1

        time.sleep(0.1)

    if n != 0:
        av_critics_grad = av_critics_grad / n
        av_actors_grad = av_actors_grad / n

    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)



















'''
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
'''








