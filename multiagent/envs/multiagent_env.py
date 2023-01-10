import gym

import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class CustomMultiAgentEnv(MultiAgentEnv):
    
    def __init__(self, config):
        #* Env
        self.dones = set()
        self.max_episode_step = config['max_episode_step']
        self.num_step = 0
        
        #* Agents
        self.agents = config['agents']
        # Set Obs and Act spaces for each agent.
        self.action_spaces = [agent.action_space
                              for agent in self.agents]
        self.observation_spaces = [agent.observation_space
                                   for agent in self.agents]

        #* Landmarks
        self.landmarks = config['landmarks']
        
        #* Rendering
        self.viewers = [None]
    
    def reset(self):
        # Reset renderer
        self._reset_render()
        for agent in self.agents:
            agent.reset()
        for landmark in self.landmarks:
            landmark.reset()
            
        self.num_step = 0

        return { agent.id: agent.observation(self) for agent in self.agents }
    
    def step(self, actions):
        obs, rew = {}, {}
        
        self.num_step += 1
        
        # Apply every agent action
        for agent in self.agents:
            agent.step(actions[agent.id], self)
        
        # Get every agent actions, observations, infos and done state
        for agent in self.agents:
            obs[agent.id] = agent.observation(self)
            rew[agent.id] = agent.reward(self)
        
        # Get agents information
        info = self.get_agents_information()
        # Check if some agents are done
        # done = self.check_agent_done()
        done = {
            "__all__": len(self.dones) == len(self.agents) or\
                self.num_step == self.max_episode_step
            }
                
        return obs, rew, done, info
    
    def get_agents_information(self):
        return { agent.id: {} for agent in self.agents }
    
    def check_agent_done(self):
        return { agent.id: False for agent in self.agents }
    
    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None
    
    def render(self, mode="human"):

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        entities = self.agents + self.landmarks
        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            pos = np.zeros(2)
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(entities):
                self.render_geoms_xform[e].set_translation(*entity.position)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results