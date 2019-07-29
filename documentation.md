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

## More Details:  
### core.py :  
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
    - How is new action of each agent determined  in each step? In this library, they use a policy class to determine agents policy. If developers are writing their own main script, the new action can be handled by any class, as long as the new action is available in the main script and gets passed to MultiAgentEnv's step() method.
    - Contains physical action of agents : self.u, which is a [float, float] list.
    - Contains communication action of agents: self.c
  - World ( All attributes are public. ) :
    - Contains a list of agents, a list of landmarks and physical aspects of this world( damping, dimension, etc. )
    - Fuctions:
      1.  ​
      2. ​
    - Relation with environment:  
