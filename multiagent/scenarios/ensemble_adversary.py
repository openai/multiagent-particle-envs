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
        world.num_agents = 3
        world.num_adversaries = 1
        num_landmarks = 2
        # add agents
        world.all_agents = [Agent() for i in range(world.total_agents)]
        world.agents = []
        for i, agent in enumerate(world.all_agents):
            agent.index = i
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = (i < world.total_adversaries)
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # randomly pick agents
        self.select_agents(world)

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.75, 0.25])
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        goal.color = np.array([0.25, 0.25, 0.75])
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        while True:
            ok=True
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                for j in range(i):
                    if np.sqrt(np.sum(np.square(world.landmarks[i].state.p_pos-world.landmarks[j].state.p_pos))) < 0.5:
                        ok=False
            if ok:
                break

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        if self.measure_success:  # return successes
            agents = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in world.agents if a.adversary == agent.adversary]
            return 1 if min(agents) <= agent.size * 2 else 0
        else:  # normal reward
            return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True
        adversary_agents = [agent for agent in world.agents if agent.adversary]
        if shaped_adv_reward:
            adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:
            adv_rew = 0
        #for a in adversary_agents:
        #    if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
        #        adv_rew -= 1
        good_agents = [agent for agent in world.agents if not agent.adversary]
        if shaped_reward:
            dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]
            pos_rew = -min(dist) #- 0.1 * max(dist)
        else:
            pos_rew = 0
            pos_rew -= min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        #if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) < 2 * agent.goal_a.size:
        #    pos_rew += 1

        # out of bound penalty
        def func(v):
            ret=0
            for p in range(world.dim_p):
                if v[p]<0: continue
                ret += np.exp(2 * v[p]) - 1
            return ret
        penalty = func(np.abs(agent.state.p_pos) - 1)
        return pos_rew + adv_rew # - penalty

    def adversary_reward(self, agent, world):
        # rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:
            adv_rew = -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
        else:
            adv_rew = 0
        if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
            adv_rew += 1
        return adv_rew
               


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
        other_to_goal = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_to_goal.append(other.state.p_pos - agent.goal_a.state.p_pos)
        if not agent.adversary:
            return np.concatenate([agent.state.p_vel] + [agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos + other_to_goal)
        else:
            other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos  # randomize position of other agents in adversary network
            return np.concatenate([agent.state.p_vel] + entity_pos + other_pos)
