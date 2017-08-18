import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import EnsembleBaseScenario
import random


class Scenario(EnsembleBaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.total_agents = 12
        world.total_adversaries = 4
        world.partition_n = 4
        world.adversary_part_n = 1
        world.good_part_n = 2
        world.num_agents = 2
        world.num_adversaries = 1
        num_landmarks = 2
        agent_per_landmark = (world.total_agents - world.total_adversaries) // num_landmarks
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # add agents
        world.all_agents = [Agent() for i in range(world.total_agents)]
        world.agents = []
        for i, agent in enumerate(world.all_agents):
            agent.index = i
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = (i < world.total_adversaries)
            if agent.adversary:
                c = 0.5 / world.total_adversaries * (i + 1) + 0.1
                agent.color = np.array([0.5 + c, 0.25, 0.25])
            else:
                x = (i - world.total_adversaries) % num_landmarks
                y = (i - world.total_adversaries) // num_landmarks
                c = 0.5 / agent_per_landmark * (y + 1) + 0.1
                agent.color = np.array([0.25, 0.25, 0.25])
                agent.color[1 + x] += c
                agent.target_landmark = x
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # randomly pick agents
        self.select_agents(world)

        assert(len(world.agents) == world.num_agents)
        assert(len(world.landmarks) == 2)
        rand_perm = np.random.permutation(len(world.landmarks))
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[rand_perm[i] + 1] = 0.9
        # set goal landmark
        goal = world.landmarks[rand_perm[world.agents[1].target_landmark]]
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        if self.measure_success:
            agents = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in world.agents]
            cur_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            #return 1 if cur_dist <= 2 * agent.size and abs(cur_dist-min(agents)) <=1e-6 else 0
            return 1 if cur_dist <= agent.size else 0
        else:
            return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # the distance to the goal
        good_agents = [agent for agent in world.agents if not agent.adversary]
        dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]
        ret = -sum(dist)
        if min(dist) <= agent.size:
            ret += 1
        #bad_agents = [agent for agent in world.agents if agent.adversary]
        #ret += 0.5 * sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in bad_agents])
        return ret

    def adversary_reward(self, agent, world):
        # keep the nearest good agents away from the goal
        good_agents = [agent for agent in world.agents if not agent.adversary]
        agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]
        pos_rew = min(agent_dist)
        #if pos_rew <= agent.size:
        #    pos_rew -= 1
        #nearest_agent = world.good_agents[np.argmin(agent_dist)]
        #neg_rew = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))
        neg_rew = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))
        #neg_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.good_agents])
        #ret = -neg_rew #pos_rew - neg_rew
        if neg_rew <= agent.size:
            neg_rew -= 1  # total reward += 1
        return pos_rew - neg_rew
               


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
        comm = []
        other_pos = []
        good_agents = [agent for agent in world.agents if not agent.adversary]
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_pos.append(other.state.p_vel)
        if not agent.adversary:
            return np.concatenate([agent.state.p_vel] + [agent.goal_a.state.p_pos - agent.state.p_pos] + [agent.color] + entity_pos + entity_color + other_pos)
        else:
            #other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos  # randomize position of other agents in adversary network
            return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)
