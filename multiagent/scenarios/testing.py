import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    goalDist = 5.0; #Currently the distance to landmark

    def make_world(self):
        world = World()	#World has agents and landmarks
        # set any world properties first
        world.dim_c = 0
        num_agents = 2      #Change this to add agents
        world.num_agents = num_agents
        num_adversaries = 0
        num_landmarks = num_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])
        for agent in world.agents:
            agent.goal_a = goal
        # set random initial states     TODO: Initialize agents + landmarks to set positions with 0 velocity
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.array([i/2,0])
            agent.state.p_vel = 0
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([i,goalDist])
            landmark.state.p_vel = 0

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # # return all adversarial agents
    # def adversaries(self, world):
    #     return [agent for agent in world.agents if agent.adversary]

    #Simplified to just distance from y = 5;
    def reward(self, agent, world):
        alpha = 0.5 #Can be adjusted to determine whether individual performance, or ranked importance is more important [0,1]
        return alpha * agent.state.p_pos[1] - (1-alpha) * 1/(world.num_agents-1)*sum([other.state.p_pos[1] for other in world.agents if other is not agent])
        #Right now + for distance - average of the distance covered by other agents.


        # return self.agent_reward(agent,world)
        # Agents are rewarded based on minimum agent distance to each landmark
        # return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    # def agent_reward(self, agent, world):   #TODO: set reward to distance to their landmark, remove adversary stuff
    #     # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
    #     shaped_reward = True
    #     shaped_adv_reward = True

    #     # # Calculate negative reward for adversary
    #     # adversary_agents = self.adversaries(world)
    #     # if shaped_adv_reward:  # distance-based adversary reward
    #     #     adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
    #     # else:  # proximity-based adversary reward (binary)
    #     #     adv_rew = 0
    #     #     for a in adversary_agents:
    #     #         if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
    #     #             adv_rew -= 5

    #     # Calculate positive reward for agents
    #     good_agents = self.good_agents(world)
    #     if shaped_reward:  # distance-based agent reward
    #         pos_rew = -min(
    #             [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
    #     else:  # proximity-based agent reward (binary)
    #         pos_rew = 0
    #         if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
    #                 < 2 * agent.goal_a.size:
    #             pos_rew += 5
    #         pos_rew -= min(
    #             [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
    #     return pos_rew + adv_rew   #Rewards are a simple int

    #Adversaries are given rewards
    # def adversary_reward(self, agent, world):
    #     # Rewarded based on proximity to the goal landmark
    #     shaped_reward = True
    #     if shaped_reward:  # distance-based reward
    #         return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
    #     else:  # proximity-based reward (binary)
    #         adv_rew = 0
    #         if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
    #             adv_rew += 5
    #         return adv_rew


    #What is passed to the agent ie How they see the world
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # entity_pos = []
        # for entity in world.landmarks:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        entity_pos = [goalDist - agent.state.p_pos[1]] #Should only need the distance to it's own landmark goal


        # communication of all other Agents
        other_pos = []
        for other in world.agents:
            # if other is agent: continue
            other_pos.append(goalDist - other.state.p_pos[1]) #Agents know how far other agents are from their goals

        if not agent.adversary:
            return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
        else:
            return np.concatenate(entity_pos + other_pos)
