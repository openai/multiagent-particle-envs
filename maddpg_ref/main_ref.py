from torch.autograd import Variable
from make_env import make_env
from gym import spaces
from MADDPG import MADDPG
import argparse
import numpy as np
import torch as th
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import time
import pdb


parser = argparse.ArgumentParser()
parser.add_argument("--consistency_interval", type=int, default=10, 
    help="Number of episodes to tally communications stats over")
parser.add_argument("-l", "--load", type=str, default=None, 
    help="Path to model to load")
parser.add_argument("--snapshot_interval", type=int, default=200,
    help="Episodes between model snapshots")
parser.add_argument("--snapshot_path", type=str, default="../snapshots/", 
    help="Path to output model snapshots")
parser.add_argument("--snapshot_prefix", type=str, default="reference_latest_episode_", 
    help="Filename prefix of model snapshots")
parser.add_argument("--print_action", action="store_true")
parser.add_argument("--print_communication", action="store_true")

args = parser.parse_args()

snapshot_interval = args.snapshot_interval
snapshot_path = args.snapshot_path
snapshot_prefix = args.snapshot_prefix
load_models = args.load
consistency_interval = args.consistency_interval
print_action = args.print_action
print_communication = args.print_communication

env = make_env('simple_reference', 
    print_action=print_action, print_communication=print_communication)
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

capacity = 30000
batch_size = 1024  # 1024

n_episode = 100000    # 20000
max_steps = 30    # 35
episodes_before_train = 50     # 50 ? Not specified in paper

maddpg = MADDPG(n_agents,
                dim_obs_list,
                dim_act_list,
                batch_size,
                capacity,
                episodes_before_train,
                action_noise="Gaussian_noise",  # ou_noises
                load_models=load_models)        # path

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

writer = SummaryWriter()

for i_episode in range(n_episode):
    '''
    # curriculum learning
    if i_episode < 1000:
        env.set_level(0)
    elif 1000 <= i_episode < 3000:
        env.set_level(1)
    else:
        env.set_level(2)
    '''
    env.set_level(2)
    obs = env.reset()
    obs = np.concatenate(obs, 0)
    if isinstance(obs, np.ndarray):
        obs = th.FloatTensor(obs).type(FloatTensor)    # obs in Tensor

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
    if (i_episode % consistency_interval) == 0:
        communication_mappings = np.zeros((n_agents, 3, 3))
    episode_communications = np.zeros((n_agents, 3))
    for t in range(max_steps):
        env.render()
        # time.sleep(0.05)

        # obs Tensor turns into Variable before feed into Actor
        obs_var = Variable(obs).type(FloatTensor)
        action = maddpg.select_action(obs_var)      # action in Variable
        action = action[0].data                     # action in Tensor
        action_np = action.cpu().numpy()            # actions in numpy array
        # convert action into list of numpy arrays
        idx = 0
        action_ls = []
        for x in dim_act_list:
            action_ls.append(action_np[idx:(idx+x)])
            idx += x
        obs_, reward, done, _ = env.step(action_ls)
        total_reward += sum(reward)
        reward = th.FloatTensor(reward).type(FloatTensor)

        comm_1 = action_np[5:8].argmax()
        comm_2 = action_np[13:16].argmax()
        episode_communications[0, comm_1] += 1
        episode_communications[1, comm_2] += 1

        obs_ = np.concatenate(obs_, 0)
        obs_ = th.FloatTensor(obs_).type(FloatTensor)

        maddpg.memory.push(obs, action, obs_, reward)  # store in Tensor

        obs = obs_

        critics_grad, actors_grad = maddpg.update_policy()

        if maddpg.episode_done > maddpg.episodes_before_train:
            av_critics_grad += np.array(critics_grad)
            av_actors_grad += np.array(actors_grad)
            n += 1
    for agent_i in range(n_agents):
        for goal_i in range(3):
            if env.world.agents[agent_i].goal_b == env.world.landmarks[goal_i]:
                communication_mappings[agent_i, goal_i, :] += episode_communications[agent_i, :]
    
    if (i_episode % consistency_interval) == consistency_interval - 1:
        for agent_i in range(n_agents):
            string = "Agent {}: ".format(agent_i)
            normalized_agent_mapping = communication_mappings[agent_i,:,:]/np.expand_dims(communication_mappings[agent_i,:,:].sum(1),1)
            writer.add_scalar('communication/agent{}_det'.format(agent_i),
                np.linalg.det(normalized_agent_mapping),
                i_episode)
            for goal_i in range(3):
                mapping = communication_mappings[agent_i,goal_i,:]
                consistency = 0 if mapping.sum() == 0 else mapping.max() / mapping.sum()
                writer.add_scalar('consistency/agent{}_goal{}'.format(agent_i, goal_i), consistency, i_episode)
                string += ("{:.1f}% ".format(consistency*100))
            print(string)

    if n != 0:
        av_critics_grad = av_critics_grad / n
        av_actors_grad = av_actors_grad / n

    maddpg.episode_done += 1
    mean_reward = total_reward / max_steps
    print('End of Episode: %d, mean_reward = %f, total_reward = %f' % (i_episode, mean_reward, total_reward))

    # plot of reward
    writer.add_scalar('data/reward_ref', mean_reward, i_episode)

    # plot of agent0 - speaker gradient of critic net
    for i in range(6):
        writer.add_scalar('data/agent0_critic_gradient', av_critics_grad[0][i], i_episode)

    # plot of agent0 - speaker gradient of actor net
    for i in range(6):
        writer.add_scalar('data/agent0_actor_gradient', av_actors_grad[0][i], i_episode)

    # plot of agent1 - listener gradient of critics net
    for i in range(6):
        writer.add_scalar('data/agent1_critic_gradient', av_critics_grad[1][i], i_episode)

    # plot of agent1 - listener gradient of critics net
    for i in range(6):
        writer.add_scalar('data/agent1_actor_gradient', av_actors_grad[1][i], i_episode)

    # to save models every N episodes
    if i_episode != 0 and i_episode % snapshot_interval == 0:
        print('Save models!')
        if maddpg.action_noise == "OU_noise":
            states = {'critics': maddpg.critics,
                      'actors': maddpg.actors,
                      'critic_optimizer': maddpg.critic_optimizer,
                      'actor_optimizer': maddpg.actor_optimizer,
                      'critics_target': maddpg.critics_target,
                      'actors_target': maddpg.actors_target,
                      'var': maddpg.var,
                      'ou_prevs': [ou_noise.x_prev for ou_noise in maddpg.ou_noises]}
        else:
            states = {'critics': maddpg.critics,
                      'actors': maddpg.actors,
                      'critic_optimizer': maddpg.critic_optimizer,
                      'actor_optimizer': maddpg.actor_optimizer,
                      'critics_target': maddpg.critics_target,
                      'actors_target': maddpg.actors_target,
                      'var': maddpg.var}
        th.save(states, snapshot_path + "/" + snapshot_prefix + str(i_episode))

writer.export_scalars_to_json("./all_scalars.json")
writer.close()






















