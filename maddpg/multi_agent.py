from make_env import make_env
import numpy as np
import time

env = make_env('simple_speaker_listener')

for i_episode in range(1):
    observation = env.reset()
    for t in range(2):
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







