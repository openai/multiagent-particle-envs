# Execution:

In a simulation with `n` agents:

1. bin/script.py loads - acts as main script
2. Loads scenario
    - `./multiagent/scenario.py.make_world()`
3. Loads multi-agent enviroment given scenario settings and world
    - `./multiagent/environment.py.MultiAgentEnv(Scenario.world())`
4. Renders environment (initial render)
    - `./multiagent/environment.py.render()`
5. Assigns policies (algorithms) for each agent
    - stored as policies[] list
    - policy[agent_index] = ./multiagent/policies/template.py.TemplatePolicy(env,agent_index)
        - Note: Template not implemented yet, see `./multiagent/policy.py.InteractivePolicy()` for now
        - For more information, see [Policies](#POLICIES)
6. Resets environment
7. Infinite while loop
    1. Makes a list of actions, one action per policy
        - actions[i]
    2. Performs one environment step using entire action list
        - `multiagent/environment.py.step()` returns:
            - n observations
            - n rewards
            - n done states
            - n debug objects
    3. Re-render
        - `multiagent/environment.py.render()`

## Environment

The main class in use during execution. The environment interacts with the scenario and the agents. There is one environment that all scenarios use. Each scenario implements reward() and observation() which the environment calls.

- `./make_env.py`: contains code for importing a multiagent environment as an OpenAI Gym-like object.

- `./multiagent/environment.py`: contains code for environment simulation (interaction physics, `_step()` function, etc.)

## Policy <a name="POLICIES"></a>

A policy seems to be a system to control an agent. The interactive policy allows control of an agent with keyboard and mouse, but if we wish to implement algorithms we will most likely be implementing them as a policy. **NOTE: Policies are enumerable**

- `./multiagent/policy.py`: contains code for interactive policy based on keyboard input.

A Policy has two functions:

- `__init__()` passes the environment to the policy class
- `action(obs)` performs an action given an observation


## Scenarios

A BaseScenario `multiagent/scenario.py` incorporates at least `make_world()` and `reset_world()`. An implemented Scenario will incorporate reward() and observation(). All scenario calls are made through the environment.

- `./multiagent/scenario.py`: contains base scenario object that is extended for all scenarios.

- `./multiagent/scenarios/`: folder where various scenarios/ environments are stored. scenario code consists of several functions:
    1) `make_world()`: creates all of the entities that inhabit the world (landmarks, agents, etc.), assigns their capabilities (whether they can communicate, or move, or both).
     called once at the beginning of each training session
    2) `reset_world()`: resets the world by assigning properties (position, color, etc.) to all entities in the world
    called before every episode (including after make_world() before the first episode)
    3) `reward(agent,world)`: defines the reward function for a given agent
    4) `observation(agent, world)`: defines the observation space of a given agent
    5) (optional) `benchmark_data()`: provides diagnostic data for policies trained on the environment (e.g. evaluation metrics)

You can create new scenarios by implementing the first 4 functions above (`make_world()`, `reset_world()`, `reward()`, and `observation()`), and have to keep the same function signature(can't not change parameters), unless we all make changes to multiagent/environment.

## Miscellaneous

- `./multiagent/core.py`: contains classes for various objects (Entities, Landmarks, Agents, etc.) that are used throughout the code.(used for creating a scenario. We might need customized entities, agents for our own scenarios.)

- `./multiagent/rendering.py`: used for displaying agent behaviors on the screen.

## Visualization:

1. Each agent will have one corresponding window generated for itself, agents always locate at the center of the camera in     its own wondow.    
2.  In the interactive policy, pressing -> will make the agent go left in the world, but everything else goes right in its  own window(since it's always at the center of its own window).  

## Implementation Details:  
### core.py :  
- Overview:  this class contains implementation of a world other scenarios are build upon. This world has 2 type of entities defined: landmarks and agents and it keeps track of each entities states. Landmarks are immovable by default, which means it will never move even a collision happens to it. Agents have 2 types of actions: physical actions and communication actions. Physical actions are forces(n-dimensional vectors represented as a list of float) agents want to exert on themselves, but their physical actions are not their final force, because collisions might happen.  Communication actions are not handled in this class and not found to be handled in other classes in the library yet.
- classes:
  - Entity ( All attributes are public. ) : 
    - Entities like agents, landmark, etc.
    - Has attributes like name, size, state, mass and some other physical aspects. 
  - Landmark( inherits Entity ) ( All attributes are public. ):
     - A type of static entity, it has no extra attributes other than those inherited from Entity, a Entity with all attributes initialized with Entity's default constructor.
  - Agent( inherits Entity ) ( All attributes are public. ) :  
     - It added some attributes Entity doesn't have, and changed some attributes' initial value. 
     - Changed attributes are: self.movable(true now) and self.sate( AgentState not EntityState )
     - Critical attributes added: self.action(see more one Action class), self.u_noise, self.c_noise and self.action_callback( used for script behavior, not used if agents' behaviors are base on policy.py )
  - EntityState <--(inherit)-- AgentState ( All attributes are public. ) :  
     - EntityState contains position and velocity of a given entity(agent, landmark, etc.) 
     - AgentState added self.c (communication) attribute. 
  - Action ( All attributes are public. ) :  
    - Action of a given agent. Action is changed  by the the _set_action method in the MultiAgentEnv class (in environment.py), and processed by the "integrate-state" method in World class. The _set_action method needs the new action to be passed to it as an argument. New actions need to be determined in the main script, and then pass to MultiAgentEnv's step method, which then calls _set_action.
    - How is new action of each agent determined  in each step? In this library, a policy class is used to determine agents policy. If developers are writing their own main script, the new action can be handled by any class, as long as the new action is available in the main script and gets passed to MultiAgentEnv's step() method.
    - Contains physical action of agents : self.u, is a list of forces on each dimension of this world ( see _set_action() method in environment.py:  agent.action.u = np.zeros ( self.world.dim_p )  ).
    - Contains communication action of agents: self.c
  - World ( All attributes are public. ) :
    - Contains a list of agents, a list of landmarks and physical aspects of this world( damping, dimension, etc. )
    - Properties: 
      entities (agents list + landmark list), policy_agents ( a list of all agents whose actions are controlled by policy) and scripted_agents (a list of all agents controlled by world's script).
    - Functions:
      1.  step(self): update all entities' physical states and agents' communication states in this world. This method is called in MultiAgentEnv.step().  It calculates forces on all entities by calling apply_action_force() and apply_environment_force(), then update all entities' physical state by calling integrate_state(), and update agents' communication states by calling update_agent_state().
      2.  apply_action_force(self, p_force): p_force is a list of forces on each entity, it updates forces on agents according to agents' physical actions and noises, and return the updated p_force list. Agents' noises are random values.
      3.  apply_environment_force(self, p_force): this method update forces on each entity according to collisions. This is done by iterating through each pair of entities and then calling get_collision_force(self, entity_a, entity_b).
      4.  get_collision_force(self, entity_a, entity_b): this method checks whether both entities collides with other entities, then it checks the distance between 2 entities and to determine whether a collision happens between these 2 entities, finally it returns forces on these 2 entities which are generated by collision. If one of the entity is not movable (landmark), this entity doesn't receive a force because of the collision even if a collision  happens.
      5.  integrate_state(self, p_force): this method updates each movable entities' velocity according to the p_force, and then update each movable entity's position based on its velocity and this world's time step(default 0.1).
      6.  update_agent_state(self, agent): this method updates a given agent's communication state according to agent's current communication action and a randomly generated noise(a list of values.). If agent is silent , agent's communication state is a list of zero values. 
    - Relation with environment:  The MultiAgentEnv class has an attribute which is a World object. The environment calls World.step() in MultiAgentEnv.step(), to update states of all entities in this world.

### environment.py

- Overview: environment.py contains the MultiAgentEnv class, which extends OpenAI gym's Env class, and has scenario functions, world, agents etc. as attributes. It's safe to just handle 2-d worlds in this class.

- functions:
  - step(self, action_n): 
    - This function updates all entities states and assign rewards to agents and returns these new state informations. It calls world.step(implemented in core.py ) and it should to be called in main script.
    - Parameters: a list of actions for each agent. 
    - Return: a list of rewards, observation, done_n and info_n.

  - reset(self):
    - It calls the scenario's reset_world function, which is used to set initial condition for the world and calls _reset_render to reset the Visualization window. It also reassign world's policy agents to self.agents. Finally it initializes each agents' observation by calling the observation function in scenario class.
    - returns: all agents' observation in a list.

  - _get_info(self, agent): returns information about a given agent. It returns nothing if info-callback is none. info_callback function is assigned to this environment object in the constructor.

  - _get_obs(self,agent): returns observation of a given agent.

  - _get_done(self, agent): unused right now.

  - _get_reward(self, agent):  
    - It calls scenario's reward() function to return the rewards a given agent  gets in a step. 
    - If the scenario's reward function isn't passed to this environment's constructor call, agent gets 0.0 as reward. 

  - _set_action(self, action, agent, action_space, time=None):

    - It sets both physical and communication action of a given agent according to parameter action. The parameter "action" is a one-dimensional list returned by policy's action() function, it has 2 parts: elements in the first part represents agent's physical action, the second part represents agent's communication action. If the world's communication dimension is 0, the second part is empty, or can be said there is no second part. If the action list has 2 part, it will be separated to 2 lists again at the beginning of this function ( "if isinstance(action_space, MultiDiscrete)..." ).

    - The first part of the action parameter can be 3 things: 5 floats or 1 integer

      - 1 integer: When the world is a 2-dimensional world and environment's "discrete_action_input" attribute is set to be true,  the agent only has 5 options : stay, move up, move down, move left and move right ( without control on the amount of force, and can't move diagonally), agent's physical is represented by  an integer among {0,1,2,3,4}. In this case, the agent can only communicate in one dimension, and this dimension is specified by the first integer in the second part of action list. 
      -  5 floats: This happens when the world is a 2-dimensional world, but agent's action can be any 2-dimensional vectors. The vector representing agent's physical action is split to 4 components on 4 "directional" axises: +x, -x, +y and -y. The components on a same axis but have opposite directions will subtract with each other, as a result these 4 components will form a 2-dimensional vector. The first float among these 4 floats is not as useful: In policy.py, this value is set to be 1.0 when the agent decides not to move, but this value is not used in environment's _set_action function at all.

    - In the final step of agent's physical action handling, the vector will be scale by the value of sensibility, which is hardcoded as 5.0.

    - No matter how agent's action is represented in the "action" parameter, in the end, agent's action.u (physical action) will be set as a list of floats, which represents a force vector. If the world's communication dimension is not 0, agent's communication action will be set as a list of floats, each float corresponds to each communication dimension.

    - In this function, agents can have continuous or discrete actions, according to some boolean attributes of this class. 

      - If actions are continuous, the given agent's physical or communicational action can be  any vectors.
      - If actions are discrete,(discrete_action_input is true)  the agent only has 5 options : stay, move up, move down, move left and move right. In this case, the action's first element is an int(0, 1,2,3 or 4 ) specifying the force vector. Other elements are for agent's communication action, and in this case, the agent can only communicate to one dimension(only one of the elements in agent.action.c list can be 1.0), and that dimension is specified by the first integer in the second part of action list ( action list has 2 part, explained in the first paragraph of this class's documentation ).
      - The actions can also be performed  "discretely",  this happens when the world class (in core.py) has attribute discrete_action and environment's force_discrete_action attribute is True. In this case, agents' final action will keep the largest component of this force vector, and set all other components 0.

      ​

## Extra Notice
### Compatiability:
The Scenario.py must be run under Linux System, the windows prompt will leads to unknown bugs
