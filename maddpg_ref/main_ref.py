from torch.autograd import Variable
from make_env import make_env
from gym import spaces
from MADDPG import MADDPG
import numpy as np
import torch as th
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import time
import pdb


env = make_env('simple_reference')
n_agents = len(env.world.agents)
dim_obs_list = [env.observation_space[i].shape[0] for i in range(n_agents)]

dim_act_list = []
for i in range(n_agents):
    if isinstance(env.action_space[i], spaces.MultiDiscrete):
        size = env.action_space[i].high - env.action_space[i].low + 1
        dim_act_list.append(sum(size))
    elif isinstance(env.action_space[i], spaces.Discrete):
        dim_act_list.append(env.action_space[i].n)
    else:
        print(env.action_space[i])

capacity = 1000000
batch_size = 2  # 1024

n_episode = 60000    # 20000
max_steps = 2    # 35
episodes_before_train = 2     # 50 ? Not specified in paper

snapshot_path = "/home/jadeng/Documents/snapshot/"
snapshot_name = "reference_latest_episode_"
path = snapshot_path + snapshot_name + '800'

maddpg = MADDPG(n_agents,
                dim_obs_list,
                dim_act_list,
                batch_size,
                capacity,
                episodes_before_train,
                load_models=None,       # path
                isOU=False)        # ou_noises

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

# writer = SummaryWriter()

for i_episode in range(n_episode):
    pdb.set_trace()
    obs = env.reset()
    obs = np.concatenate(obs, 0)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()    # obs in Tensor now
    total_reward = 0.0
    av_critics_grad = np.zeros((n_agents, 6))
    av_actors_grad = np.zeros((n_agents, 6))
    n = 0
    print('Simple Reference')
    print('Start of episode', i_episode)
    print("Target landmark for agent 0: {}, Target landmark color: {}"
          .format(env.world.agents[1].goal_b.name, env.world.agents[1].goal_b.color))
    print("Target landmark for agent 1: {}, Target landmark color: {}"
          .format(env.world.agents[0].goal_b.name, env.world.agents[0].goal_b.color))
    for t in range(max_steps):
        # print(t)
        env.render()

        # obs Tensor turns into Variable before feed into Actor
        obs = Variable(obs).type(FloatTensor)
        # print('obs', obs)

        action = maddpg.select_action(obs).data.cpu()   # actions in Variable
        # convert action from Variable to list, hard code here
        action = [action[0].numpy()[:dim_act_list[0]], action[0].numpy()[dim_act_list[0]:]]
        obs_, reward, done, _ = env.step(action)

        action = np.concatenate(action, 0)
        action = th.from_numpy(action).float()
        # print('action', action)

        reward = th.FloatTensor(reward).type(FloatTensor)

        # obs_ = [obs_[i] for i in range(n_agents)]
        obs_ = np.concatenate(obs_, 0)
        obs_ = th.from_numpy(obs_).float()

        '''
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None
        '''

        total_reward += reward.sum()

        maddpg.memory.push(obs.data, action, obs_, reward)  # tensors
        # print('obs', obs.data)
        # print('action', action)
        # print('next_obs', next_obs)
        # print('reward', reward)

        obs = obs_

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
    print('End of Episode: %d, mean_reward = %f, total_reward = %f' % (i_episode, mean_reward, total_reward))
    # reward_record.append(total_reward)

    '''
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




















