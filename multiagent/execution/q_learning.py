from collections import defaultdict
import numpy as np
from multiagent.policy.epsilon_greedy import EpsilonGreedyPolicy
from multiagent.algorithm.q_learning_update import q_learning_update

def q_learning_execution(env):
    # q learning initializations

    # dictionary that maps from state, s, to a numpy array of Q values [Q(s, a_1), Q(s, a_2) ... Q(s, a_n)] and everything is initialized to 0.
    q_vals = defaultdict(lambda: np.array([0. for _ in range(4)]))  # env.action_space.n = 4
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.5

    # table with velocity, agents pos, dist between agent and landmark
    current_state = env.reset()

    '''
    def greedy_eval():
        # evaluate greedy policy w.r.t current q_vals
        test_env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)
        prev_state = test_env.reset()
        ret = 0.
        done = False
        H = 100
        for i in range(H):
            action = np.argmax(q_vals[prev_state])
            state, reward, done, info = test_env.step(action)
            ret += reward
            prev_state = state
        return ret / H
    '''


    # create policies for each agent
    # env.n = number of agents
    policies = [EpsilonGreedyPolicy(env, i) for i in range(env.n)]

    # for i_episode in range(300000):
    for i_episode in range(20):
        # select action using eps_greedy
        # query for action from each agent's policy
        action = []
        for i, policy in enumerate(policies):
            action.append(policy.action(tuple(current_state[i]), q_vals, epsilon))

        # print('CurrentState - ', tuple(current_state))
        # run action
        next_state, reward, done, info = env.step(action)
        # print('Next State - ',next_state)

        # update q value
        for i in range(env.n):
            q_learning_update(gamma, alpha, q_vals, tuple(current_state[i]), action, tuple(next_state[i]), reward)

        current_state = next_state

        # render all agent views
        env.render()

        # display rewards
        for agent in env.world.agents:
            print(agent.name + " reward: %0.3f" % env._get_reward(agent))

        # evaluation
        # if i_episode % 50000 == 0: 
        #     print("Episode %i # Average reward: %.2f" % (i_episode, greedy_eval()))