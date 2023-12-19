from make_env import make_env
import numpy as np
import time
env = make_env(scenario_name='simple_survey_region')
env.reset()
print(env.agents)
while True:
    action_n = np.random.random(size=(3,7))
    # print(action_n)
    obs, rew, done, info = env.step(action_n)
    env.render()
    # print(obs)
    time.sleep(0.1)
    # print(obs)