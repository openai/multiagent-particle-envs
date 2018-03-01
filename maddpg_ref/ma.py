from make_env import make_env
import numpy as np
import time

env = make_env('simple_reference')

for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        agent_actions = []
        print(t)
        for i, agent in enumerate(env.world.agents):
            agent_action_space = env.action_space[i]
            action = agent_action_space.sample()
            print(action)

            high = agent_action_space.high
            size = agent_action_space.high - agent_action_space.low + 1
            action_vec = np.zeros(sum(size))

            length = len(size)
            for n in range(length):
                action_vec[sum(size[:n]) + action[n]] = 1
            agent_actions.append(action_vec)
            print(action_vec)

        time.sleep(0.1)
        import pdb
        pdb.set_trace()
        observation, reward, done, info = env.step(agent_actions)

        print(observation)
        print(reward)
        print(done)
        print(info)
        print('env.world.agents[0].action.u: ', env.world.agents[0].action.u)
        print('env.world.agents[0].action.c: ', env.world.agents[0].action.c)
        print('env.world.agents[1].action.u: ', env.world.agents[1].action.u)
        print('env.world.agents[1].action.c: ', env.world.agents[1].action.c)


'''
# try out discrete input instead of one hot encoding,
# agents not moving
print('discrete_action_input = True')
env1 = make_env('simple_reference')
env1.discrete_action_input = True

for i_episode in range(1):
    observation = env1.reset()
    for t in range(5):
        env1.render()
        agent_actions = []
        for i, agent in enumerate(env1.world.agents):
            agent_action_space = env1.action_space[i]
            action = agent_action_space.sample()
            print(action)

            agent_actions.append(action)

        print(agent_actions)

        time.sleep(1)
        observation, reward, done, info = env1.step(agent_actions)

        print(observation)
        print(reward)
        print(done)
        print(info)
        print()
'''
