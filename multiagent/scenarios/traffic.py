import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_agents = 2
        num_adversaries = 0
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def sample_z_normal(self, mean=0, std=1, size=4, batch_size=1):
        return np.squeeze(np.random.normal(mean, std, size=(batch_size, size)))

    def reset_world(self, world):
        world.timer = 0
        world.shared_var = self.sample_z_normal()
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([1.0, 1.0, 1.0])
            landmark.size = 0.0
            landmark.index = i

        # landmark initial states
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.state.p_pos = np.asarray([0.0, -1.0])
                landmark.state.p_vel = np.zeros(world.dim_p)
            elif i == 1:
                landmark.state.p_pos = np.asarray([-1.0, 0.0])
                landmark.state.p_vel = np.zeros(world.dim_p)

        # Agent properties
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            agent.size = 0.35
        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.goal = world.landmarks[i]
            if i == 0:
                # agent.state.p_pos = np.asarray([0.0, 1.0])
                # Random position
                offset = np.random.exponential(scale=1 / 5)
                if offset > 0.2:
                    offset = 0.2
                agent.state.p_pos = np.asarray([0.0, 1.0 - offset])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            elif i == 1:
                # agent.state.p_pos = np.asarray([1.0, 0.0])
                offset = np.random.exponential(scale=1 / 5)
                if offset > 0.2:
                    offset = 0.2
                agent.state.p_pos = np.asarray([1.0 - offset, 0.0])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        tol = 0.1
        dist_to_goal = np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos))
        reward = 0
        reached_goal = False

        if self.is_collision(world.agents[0], world.agents[1]):
            reward -= 200

        # Agent 0 is going vertically (top to bottom)
        if int(agent.name.split(" ")[-1]) == 0:
            if np.abs(agent.state.p_pos[0]) > tol:
                reward -= 200

            if agent.state.p_pos[1] <= agent.goal.state.p_pos[1]:
                reward += 10
                reached_goal = True

        # Agent 1 is going horizontally (right to left)
        else:
            if np.abs(agent.state.p_pos[1]) > tol:
                reward -= 200
            if agent.state.p_pos[0] <= agent.goal.state.p_pos[0]:
                reward += 10
                reached_goal = True

        if not reached_goal:
            reward -= 1.0 + dist_to_goal

        return reward

    def observation(self, agent, world):
        goal_pos = agent.goal.state.p_pos - agent.state.p_pos
        return np.concatenate(
            [agent.state.p_vel, agent.state.p_pos, goal_pos, [world.timer]]
        )

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_done(self, agent, world):
        if self.is_collision(world.agents[0], world.agents[1]):
            return True
        else:
            return False
