import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import EnsembleBaseScenario
import random


class Scenario(EnsembleBaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.total_agents = 15
        world.total_adversaries = 3
        world.partition_n = 3  # now only support partition_n <= 3
        world.adversary_part_n = 1
        world.good_part_n = 4
        world.num_agents = 5
        world.num_adversaries = 1
        num_landmarks = 1
        landmark_size = 0.2
        entity_size = 0.05
        chaser_size = 0.075
        self.colide_dist = entity_size + chaser_size + 1e-2
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.size = landmark_size
            landmark.collide = True
            landmark.movable = False
            landmark.color = np.array([0.25, 0.25, 0.25])
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
                agent.size = entity_size
                agent.accel = 6.0
                c = 0.5 / world.total_adversaries * (i + 1)
                agent.color = np.array([0.5 + c, 0.25, 0.25])
            else:
                agent.size = chaser_size
                agent.accel = 3.0
                team_size = world.num_agents - world.num_adversaries
                x = (i - world.total_adversaries) % team_size
                y = (i - world.total_adversaries) // team_size
                c = 0.5 / team_size * (x + 1) + 0.1
                agent.color = np.array([0.25, 0.25, 0.25])
                if y < 2:
                    agent.color[1 + y] += c
                else:
                    agent.color[1] += c
                    agent.color[2] += c
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # randomly pick agents
        self.select_agents(world)

        assert(len(world.agents) == world.num_agents)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        for i, agent in enumerate(world.agents):
            while True:
                if agent.adversary:
                    agent.state.p_pos = np.random.uniform(-0.5, 0.5, world.dim_p)
                else:
                    agent.state.p_pos = np.random.uniform(-1, -0.5, world.dim_p)
                    for p in range(world.dim_p):
                        if np.random.uniform(0,1) < 0.5:
                            agent.state.p_pos[p] *= -1
                ok=True
                for j in range(i):
                    b = world.agents[j]
                    dist = np.sqrt(np.sum(np.square(agent.state.p_pos - b.state.p_pos)))
                    if dist < agent.size + b.size + 0.01:
                        ok=False
                        break
                for b in world.landmarks:
                    dist = np.sqrt(np.sum(np.square(agent.state.p_pos - b.state.p_pos)))
                    if dist < agent.size + b.size + 0.01:
                        ok = False
                        break
                if ok: break
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        if self.measure_success:
            good_agents = [a for a in world.agents if not a.adversary]
            bad_agents = [a for a in world.agents if a.adversary]
            bad_agent = bad_agents[0]  # assume only one adversary
            allow_dist = self.colide_dist
            if agent.adversary:
                return -1 if min([np.sqrt(np.sum(np.square(bad_agent.state.p_pos - a.state.p_pos))) for a in good_agents]) <= allow_dist else 0
            else:
                return 1 if np.sqrt(np.sum(np.square(agent.state.p_pos - bad_agent.state.p_pos))) <= allow_dist else 0
        else:
            return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # want to catch the agent
        bad_agents = [agent for agent in world.agents if agent.adversary]
        good_agents = [agent for agent in world.agents if not agent.adversary]
        rew = 0
        catch_reward = 0
        for b in bad_agents:
            dist = [np.sqrt(np.sum(np.square(b.state.p_pos - a.state.p_pos))) for a in good_agents]
            if min(dist) <= self.colide_dist:
                catch_reward += 5
            rew += sum(dist)
        return -rew + catch_reward

    def adversary_reward(self, agent, world):
        # keep self in the court and keep away from the
        good_agents = [agent for agent in world.agents if not agent.adversary]
        def func(p):
            # return x * (1 + x)
            return np.square(p) #p + np.square(p)
        agent_dist = [np.sqrt(np.sum(func(agent.state.p_pos - a.state.p_pos))) for a in good_agents]
        pos_rew = min(agent_dist)
        if pos_rew <= self.colide_dist:
            pos_rew -= 5
        #nearest_agent = world.good_agents[np.argmin(agent_dist)]
        neg_rew = 0
        def func(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2*x-2), 10) #1 + (x - 1) * (x - 1)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            neg_rew += func(x)
            
        #neg_rew = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))
        #neg_rew = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))
        #neg_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.good_agents])
        return pos_rew - neg_rew
               


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        good_agents = [agent for agent in world.agents if not agent.adversary]
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_pos.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel, agent.state.p_pos-1, agent.state.p_pos+1] + entity_pos + other_pos)
