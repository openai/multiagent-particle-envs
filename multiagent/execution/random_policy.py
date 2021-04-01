from multiagent.policy.random import RandomPolicy

def random_policy_execution(env):
    policies = [RandomPolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()

    # while True:
    for i_episode in range(20):
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))

        print('Current State - ',obs_n)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        print('Next State - ',obs_n)

        # render all agent views
        env.render()
        
        # display rewards
        for agent in env.world.agents:
           print(agent.name + " reward: %0.3f" % env._get_reward(agent))