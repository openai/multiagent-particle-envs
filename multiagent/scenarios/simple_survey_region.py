import numpy as np
from multiagent.core import World, Agent, Obstacle
from multiagent.scenario import BaseScenario


class SurveyScenario(BaseScenario):
    def __init__(self, num_obstacles, num_agents, vision_dist, grid_resolution, grid_max_reward, reward_delta, observation_mode):
        self.num_obstacles = num_obstacles
        self.num_agents = num_agents
        self.vision_dist = vision_dist
        self.grid_resolution = grid_resolution
        self.grid_max_reward = grid_max_reward
        self.reward_delta = reward_delta
        self.observation_mode = observation_mode

    def make_world(self):
        world = World()
        world.dim_c = 2
        world.collaborative = False

        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.vision_dist = self.vision_dist

        # Initialize grid
        world.grid = np.zeros((self.grid_resolution, self.grid_resolution))
        world.obstacles = [self._create_random_obstacle(i) for i in range(self.num_obstacles)]
        world.obstacle_mask = self._create_obstacle_mask(world)
        world.reward_mask = self._create_reward_mask()

        # make initial conditions
        self.reset_world(world)
        return world
    
    def _create_random_obstacle(self, i):
        obstacle = Obstacle()
        obstacle.name = 'obstacle {}'.format(i)
        obstacle.collide = True
        obstacle.color = np.array([1.0, 0.0, 0.0])  # Red color

        # Grid resolution
        grid_resolution = self.grid_resolution

        # Random start grid square
        start_x = np.random.randint(0, grid_resolution)
        start_y = np.random.randint(0, grid_resolution)

        # Random length (1 to 4 grid squares)
        length = np.random.randint(1, 5)

        # Random direction (0 for horizontal, 1 for vertical)
        direction = np.random.randint(0, 2)

        # Initialize the obstacle mask
        obstacle_mask = np.zeros((grid_resolution, grid_resolution))

        # Create the obstacle based on direction
        if direction == 0:  # Horizontal
            end_x = min(start_x + length, grid_resolution)
            for x in range(start_x, end_x):
                obstacle_mask[x, start_y] = 1
            obstacle.width = (end_x - start_x) * (2 / grid_resolution)
            obstacle.height = 2 / grid_resolution
        else:  # Vertical
            end_y = min(start_y + length, grid_resolution)
            for y in range(start_y, end_y):
                obstacle_mask[start_x, y] = 1
            obstacle.width = 2 / grid_resolution
            obstacle.height = (end_y - start_y) * (2 / grid_resolution)

        # Set the position of the obstacle (center)
        obstacle.state.p_pos = np.array([
            (start_x + obstacle.width / (2 * (2 / grid_resolution))) * (2 / grid_resolution) - 1,
            (start_y + obstacle.height / (2 * (2 / grid_resolution))) * (2 / grid_resolution) - 1
        ])

        # Store the mask in the obstacle for later use
        obstacle.mask = obstacle_mask

        return obstacle
    def _create_reward_mask(self, zeros_count=10):
        # Initialize the reward mask with ones
        reward_mask = np.ones((self.grid_resolution, self.grid_resolution))
        
        # Randomly choose grid squares to be zero
        zero_indices = np.random.choice(self.grid_resolution * self.grid_resolution, zeros_count, replace=False)
        
        # Convert flat indices to 2D indices and assign zero
        for index in zero_indices:
            x, y = divmod(index, self.grid_resolution)
            reward_mask[x, y] = 0
        
        return reward_mask
    def _create_obstacle_mask(self, world):
        # Initialize the grid mask with ones
        obstacle_mask = np.ones((self.grid_resolution, self.grid_resolution))

        # Combine the individual masks of each obstacle
        for obstacle in world.obstacles:
            obstacle_mask = np.minimum(obstacle_mask, 1 - obstacle.mask)

        return obstacle_mask
    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            # Initialize agent position
            self.initialize_agent_position(agent, world)
        
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            # Random position for landmarks
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def initialize_agent_position(self, agent, world):
        while True:
            # Generate a random position for the agent
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_angle = np.random.uniform(0, 2*np.pi, 1)
            agent.state.p_angle_vel = np.zeros(1)
            # Check for collision with any obstacle
            collision = any(
                self.is_collision_rectangular(agent, obstacle, agent.state.p_pos)
                for obstacle in world.obstacles
            )

            # If no collision, break the loop
            if not collision:
                break

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        def get_grid_coord(pos):
            """Converts a position to grid coordinates."""
            return min(int((pos + 1) / 2 * self.grid_resolution), self.grid_resolution - 1)

        def get_line(start, end):
            """Generate the points of a line using Bresenham's algorithm."""
            # Setup initial conditions
            x1, y1 = start
            x2, y2 = end
            dx = x2 - x1
            dy = y2 - y1

            is_steep = abs(dy) > abs(dx)  # Determine how steep the line is

            # Rotate line if steep
            if is_steep:
                x1, y1 = y1, x1
                x2, y2 = y2, x2

            # Swap start and end points if necessary
            swapped = False
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                swapped = True

            dx = x2 - x1
            dy = y2 - y1

            error = dx / 2.0
            ystep = 1 if y1 < y2 else -1

            y = y1
            points = []
            for x in range(x1, x2 + 1):
                coord = (y, x) if is_steep else (x, y)
                points.append(coord)
                error -= abs(dy)
                if error < 0:
                    y += ystep
                    error += dx

            if swapped:
                points.reverse()

            return points

        # Calculate start and end points in grid coordinates
        start_x = get_grid_coord(agent.state.p_pos[0])
        start_y = get_grid_coord(agent.state.p_pos[1])
        end_x = get_grid_coord(agent.state.p_pos[0] + agent.vision_dist * np.cos(agent.state.p_angle))
        end_y = get_grid_coord(agent.state.p_pos[1] + agent.vision_dist * np.sin(agent.state.p_angle))

        reward = world.grid[start_x, start_y]  # Initialize reward
        world.grid[start_x, start_y] = 0  # Clear the agent's current square

        # Use Bresenham's algorithm to accurately determine the line of sight
        line_points = get_line((start_x, start_y), (end_x, end_y))

        for (x, y) in line_points:
            if 0 <= x < self.grid_resolution and 0 <= y < self.grid_resolution:
                if world.obstacle_mask[x, y] == 0:
                    break  # Line of sight is blocked by an obstacle

                # Accumulate reward and clear the grid square
                reward += world.grid[x, y]
                world.grid[x, y] = 0

        # Update the reward grid for the last agent, if necessary
        if agent == world.agents[-1]:
            grid_update = self.reward_delta * np.ones(shape=(self.grid_resolution, self.grid_resolution))
            world.grid += grid_update
            world.grid = np.clip(world.grid, a_min=0, a_max=self.grid_max_reward)
            # Don't reward for obstacle grid squares
            world.grid *= world.obstacle_mask
            # And, don't reward for masked grid squares
            world.grid *= world.reward_mask

        return reward


    def is_collision_rectangular(self, agent, obstacle, new_pos):
        ax, ay = new_pos
        agent_radius = agent.size
        ox, oy = obstacle.state.p_pos
        half_width, half_height = obstacle.width / 2, obstacle.height / 2

        left_bound = ox - half_width - agent_radius
        right_bound = ox + half_width + agent_radius
        bottom_bound = oy - half_height - agent_radius
        top_bound = oy + half_height + agent_radius

        return (left_bound <= ax <= right_bound) and (bottom_bound <= ay <= top_bound)
    
    def _get_img_obs(self, agent, world):
        """
        Generates an image-based observation for the given agent with a global view.

        The observation is a grid of size (self.grid_resolution, self.grid_resolution) with the following channels:
        - Agent's position: Binary grid indicating the agent's current position.
        - Agent's field of vision: Binary grid indicating the squares within the agent's field of vision.
        - Other agents' positions: Binary grid indicating the positions of other agents.
        - Other agents' fields of vision: Binary grid indicating the squares within other agents' fields of vision.
        - Obstacles: Binary grid indicating the locations of obstacles.
        - Reward values: Grid containing the reward values of each grid square.
        - Reward mask: Binary grid indicating the grid squares where rewards are available.
        """
        # Initialize the observation grid with zeros
        obs_grid = np.zeros((self.grid_resolution, self.grid_resolution, 7))

        # Set the agent's position channel
        agent_x, agent_y = self._pos_to_grid(agent.state.p_pos)
        obs_grid[agent_x, agent_y, 0] = 1

        # Set the agent's field of vision channel
        for x in range(max(0, agent_x - int(agent.vision_dist * self.grid_resolution / 2)),
                    min(self.grid_resolution, agent_x + int(agent.vision_dist * self.grid_resolution / 2) + 1)):
            for y in range(max(0, agent_y - int(agent.vision_dist * self.grid_resolution / 2)),
                        min(self.grid_resolution, agent_y + int(agent.vision_dist * self.grid_resolution / 2) + 1)):
                obs_grid[x, y, 1] = 1

        # Set the other agents' positions and fields of vision channels
        for other in world.agents:
            if other is not agent:
                other_x, other_y = self._pos_to_grid(other.state.p_pos)
                obs_grid[other_x, other_y, 2] = 1
                for x in range(max(0, other_x - int(other.vision_dist * self.grid_resolution / 2)),
                            min(self.grid_resolution, other_x + int(other.vision_dist * self.grid_resolution / 2) + 1)):
                    for y in range(max(0, other_y - int(other.vision_dist * self.grid_resolution / 2)),
                                min(self.grid_resolution, other_y + int(other.vision_dist * self.grid_resolution / 2) + 1)):
                        obs_grid[x, y, 3] = 1

        # Set the obstacles channel
        obs_grid[:, :, 4] = 1 - world.obstacle_mask

        # Set the reward values channel
        obs_grid[:, :, 5] = world.grid

        # Set the reward mask channel
        obs_grid[:, :, 6] = world.reward_mask

        return obs_grid


    def _pos_to_grid(self, pos):
        """
        Converts a position in the environment to a grid coordinate.
        """
        grid_x = int((pos[0] + 1) / 2 * self.grid_resolution)
        grid_y = int((pos[1] + 1) / 2 * self.grid_resolution)
        return grid_x, grid_y

    def _get_dense_obs(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [agent.state.p_angle] + [agent.state.p_angle_vel] + entity_pos + other_pos)


    def observation(self, agent, world):
        if self.observation_mode == "dense":
            return self._get_dense_obs(agent, world)
        elif self.observation_mode == "image":
            return self._get_img_obs(agent, world)
        else:
            raise ValueError("Invalid observation mode selected. Please set this parameter to 'dense' or 'image'.")





