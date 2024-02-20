import numpy as np
import time
from multiagent.survey_environment import SurveyEnv
from multiagent.utils.visualization import visualize_image_observation
# Change to have all of the configurable items out of the scenario
# Would be good to have a configuration loader method as well

env = SurveyEnv(num_agents=3, num_obstacles=4, vision_dist=0.2, grid_resolution=10, grid_max_reward=1, reward_delta=0.001, observation_mode="image")
env.reset()
print(env.agents)

while True:
    # first dimension in action space is number of agents, second is action space
    action_n = np.random.random(size=(3,7))
    obs, rew, done, info = env.step(action_n)
    env.render()
    # visualize_image_observation(obs[0])
    time.sleep(0.1)