import numpy as np
from multiagent.coreRace import World, Agent, Landmark
from multiagent.scenario import BaseScenario

backWardPunishment = 5


class Scenario(BaseScenario):
    def __init__(self):
        super(Scenario, self).__init__()
        self.agentsToLandMarks = {}
        self.numberOfAgents=6

    def make_world(self):
        world = World()
        # add agents

        world.agents = [Agent() for i in range(self.numberOfAgents)]
        for i, agent in enumerate(world.agents):
            print(i)
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.numberOfAgents)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        #fill in the dictionary
        for i in range(self.numberOfAgents):
        	self.agentsToLandMarks.update({ world.agents[i]: world.landmarks[i] })

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.25,0.25])
        world.landmarks[0].color = np.array([0.75,0.25,0.25])
        # set random initial states
        for i,agent in enumerate(world.agents):
            # agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_pos = np.array([i/self.numberOfAgents,0])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_pos = np.array([i/self.numberOfAgents,0])
            landmark.state.p_vel = np.zeros(world.dim_p)


    #Rongyi Zhang(zrysnd): reward function handles all agents, agent parameter is useless.
    #Keeping the agent here because this function is inherited.
    def reward(self, agent, world):
        '''
        cheat: any force more than 0.0
        if more than half or half of the agents decides to cheat-> cheat, cheat
        elif all agents cooperate(0.0) -> cooperate, cooperate, both move forward.
        else: some agents cheat, but less than half of the number of agents -> cheat, cooperate. cheaters move forward,
        agent.action.u = [float1, float2], float 2: up(+)/down(-)
        cheat-cheat: all stay, all cooperate: all move as 1.0, cheat cooperate: cheater moves more.
        '''


        def agent_cheated(agent):
            if agent.action.u[1] > 0.0:
                return True
            return False

        reward_n = []
        numOfCheaters = 0
        numOfAgents = 0
        for i, thisAgent in enumerate(world.agents):
            # if i == 0:
            # print(i)
            # print(agent.action.u)
            numOfAgents += 1
            thisAgent.action.u[0] = 0.0 #invalidate horizontal action
            if agent_cheated(thisAgent): #it's trying to move up
                numOfCheaters += 1
        # print(numOfCheaters)

        for i, thisAgent in enumerate(world.agents):
            if numOfCheaters > numOfAgents//2:
                thisAgent.action.u[1] = 0.0 # all cheat: stay
            elif numOfCheaters == 0:
                thisAgent.action.u[1] = 0.1 # all cooperate: move up together
            else:
                if thisAgent.action.u[1] > 0.0: # this agent cheat
                    thisAgent.action.u[1] += 0.2
                else:
                    thisAgent.action.u[1] = 0.0
            reward_n.append(thisAgent.action.u[1])
        return reward_n

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)

