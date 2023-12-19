import numpy as np
from multiagent.core import World, Agent, Landmark, Obstacle
from multiagent.scenario import BaseScenario
import math

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        # Create and add obstacles
        num_obstacles = 4 # configurable

        world.dim_c = 2
        num_agents = 3 # configurable
        world.collaborative = False # configurable
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.vision_dist = 0.2 # configurable

        # Initialize grid
        self.grid_resolution = 10  # configurable
        # The maximum reward of a given grid square
        self.grid_max_reward = 1  # configurable
        # The amount to change reward of a given grid cell after each time step
        self.reward_delta = 0.0001 # configurable
        world.grid = np.zeros((self.grid_resolution, self.grid_resolution))
        world.obstacles = [self.create_random_obstacle(i) for i in range(num_obstacles)]
        world.obstacle_mask = self.create_obstacle_mask(world)
        # make initial conditions
        self.reset_world(world)
        return world
    
    def create_random_obstacle(self, i):
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

    def create_obstacle_mask(self, world):
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
            return min(int((pos + 1) / 2 * self.grid_resolution), self.grid_resolution - 1)

        # Calculate grid square based on agent position
        grid_x = get_grid_coord(agent.state.p_pos[0])
        grid_y = get_grid_coord(agent.state.p_pos[1])
        grid_square = world.grid[grid_x, grid_y]
        world.grid[grid_x, grid_y] = 0

        # Calculate reward for squares in the agent's line of sight
        angle = agent.state.p_angle
        for dist in np.linspace(0, agent.vision_dist, num=int(agent.vision_dist * self.grid_resolution)):
            # Calculate the coordinates along the line of sight
            sight_x = agent.state.p_pos[0] + dist * np.cos(angle)
            sight_y = agent.state.p_pos[1] + dist * np.sin(angle)

            sight_grid_x = get_grid_coord(sight_x)
            sight_grid_y = get_grid_coord(sight_y)

            # Check if the line of sight is obstructed by an obstacle
            if 0 <= sight_grid_x < self.grid_resolution and 0 <= sight_grid_y < self.grid_resolution:
                if world.obstacle_mask[sight_grid_x, sight_grid_y] == 0:
                    # Line of sight is blocked by an obstacle
                    break  # Stop checking further squares in this direction

                # Add grid value to reward and reset grid square
                grid_square += world.grid[sight_grid_x, sight_grid_y]
                world.grid[sight_grid_x, sight_grid_y] = 0

        # Update reward grid if processing last agent
        if agent == world.agents[-1]:
            grid_update = self.reward_delta * np.ones(shape=(self.grid_resolution, self.grid_resolution))

            world.grid += grid_update
            world.grid = np.clip(world.grid, a_min=0, a_max=self.grid_max_reward)
            world.grid *= world.obstacle_mask

        # Return accumulated grid value as reward
        return grid_square

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
    def observation(self, agent, world):
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
