
from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.simple_survey_region import SurveyScenario

class SurveyEnv(MultiAgentEnv):
    def __init__(self, num_obstacles: int = 4, num_agents: int = 3, vision_dist: float = 0.5, grid_resolution: int = 10, grid_max_reward: float = 1.0, reward_delta: float = 0.0001, observation_mode: str = "image"):
        """
        Initializes the Survey environment with the specified configuration.

        Parameters:
            num_obstacles (int): The number of obstacles to be placed in the environment. These obstacles obstruct the agents' paths and limit their ability to navigate and view grid squares.
            num_agents (int): The number of agents to be included in the environment. Each agent can interact with the environment.
            vision_dist (float): The maximum distance at which an agent can perceive its surroundings. This affects the length of the 'red line' in the rendered simulation.
            grid_resolution (int): The resolution of the grid used to represent the environment. A higher resolution means a finer grid with more cells.
            grid_max_reward (float): The maximum reward value that a grid cell can have. This value is used to initialize the reward distribution in the environment, and to cap reward from increasing indefinitely.
            reward_delta (float): The amount by which the reward of a grid cell increases after each time step. This can be used to control the frequency with which an optimal policy would dictate than an agent should re-visit a grid square.
            observation_mode (str): The mode of observation for the agents. This can determine how agents perceive their environment, e.g., as raw pixel values ("image") or as processed features ("dense").

        Returns:
            None
        """
        # Load the scenario with the specified parameters
        self.scenario = SurveyScenario(num_obstacles, num_agents, vision_dist, grid_resolution, grid_max_reward, reward_delta, observation_mode)
        # Create the world
        world = self.scenario.make_world()
        # Initialize the parent class with the necessary functions
        super().__init__(world, self.scenario.reset_world, self.scenario.reward, self.scenario.observation)