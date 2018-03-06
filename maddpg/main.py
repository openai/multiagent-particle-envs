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
batch_size = 1024

n_episode = 25000    # 20000
max_steps = 250    # 1000
episodes_before_train = 50     # 50 ? Not specified in paper
episodes_to_break = 500

# reward_record = []

snapshot_path = "/home/jadeng/Documents/snapshot/"
snapshot_name = "speaker_listener_latest_episode_"
path = snapshot_path + snapshot_name + '800'

maddpg = MADDPG(n_agents, dim_obs_list, dim_act_list, batch_size, capacity, episodes_before_train, load_models=None)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

writer = SummaryWriter()

for i_episode in range(n_episode):
    obs = env.reset()
    # obs = [obs[i] for i in range(n_agents)]
    # import pdb
    # pdb.set_trace()
    obs = np.concatenate(obs, 0)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()    # obs in Tensor now
    total_reward = 0.0
    av_critics_grad = np.zeros((n_agents, 6))
    av_actors_grad = np.zeros((n_agents, 6))
    n = 0
    print('Start of episode', i_episode)
    print('Target landmark for agent 1: ', env.world.agents[0].goal_b.name)
    print('Target landmark color: ', env.world.agents[0].goal_b.color)
    for t in range(max_steps):
        # print(t)
        env.render()

        # obs turns to Variable before feed into Actor
        obs = Variable(obs).type(FloatTensor)
        # print('obs', obs)

        action = maddpg.select_action(obs).data.cpu()   # actions in Variable
        # convert action from Variable to list
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
        '''
        if i_episode >= episodes_to_break and reward.sum() > 5.0:
            break
        '''
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

        # time.sleep(0.05)

    if n != 0:
        av_critics_grad = av_critics_grad / n
        av_actors_grad = av_actors_grad / n

    maddpg.episode_done += 1
    mean_reward = total_reward / max_steps
    '''
    import pdb
    pdb.set_trace()
    if i_episode >= episodes_to_break and n < max_steps:
        mean_reward = total_reward / n
    else:
        mean_reward = total_reward / max_steps
    '''

    print('End of Episode: %d, mean_reward = %f, total_reward = %f' % (i_episode, mean_reward, total_reward))
    # reward_record.append(total_reward)

    # plot of reward
    writer.add_scalar('data/reward', mean_reward, i_episode)

    # plot of agent0 - speaker gradient of critic net
    for i in range(6):
        writer.add_scalar('data/speaker_critic_gradient', av_critics_grad[0][i], i_episode)

    # plot of agent0 - speaker gradient of actor net
    for i in range(6):
        writer.add_scalar('data/speaker_actor_gradient', av_actors_grad[0][i], i_episode)

    # plot of agent1 - listener gradient of critics net
    for i in range(6):
        writer.add_scalar('data/listener_critic_gradient', av_critics_grad[1][i], i_episode)

    # plot of agent0 - speaker gradient of critics net
    for i in range(6):
        writer.add_scalar('data/listener_actor_gradient', av_actors_grad[1][i], i_episode)

    # to save models every 200 episodes
    if i_episode != 0 and i_episode % 200 == 0:
        print('Save models!')
        states = {'critics': maddpg.critics,
                  'actors': maddpg.actors,
                  'critic_optimizer': maddpg.critic_optimizer,
                  'actor_optimizer': maddpg.actor_optimizer,
                  'critics_target': maddpg.critics_target,
                  'actors_target': maddpg.actors_target,
                  'memory': maddpg.memory,
                  'var': maddpg.var}
        th.save(states, snapshot_path + snapshot_name + str(i_episode))

# print('reward_record', reward_record)

writer.export_scalars_to_json("./all_scalars.json")
writer.close()



















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








