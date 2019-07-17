# Environment

- `make_env.py`: contains code for importing a multiagent environment as an OpenAI Gym-like object.

- `./multiagent/environment.py`: contains code for environment simulation (interaction physics, `_step()` function, etc.)

# Policy

A policy seems to be a system to control an agent. The interactive policy allows control of an agent with keyboard and mouse, but if we wish to implement algorithms we will most likely be implementing them as a policy.

- `./multiagent/policy.py`: contains code for interactive policy based on keyboard input.


# Scenarios

- `./multiagent/scenario.py`: contains base scenario object that is extended for all scenarios.

- `./multiagent/scenarios/`: folder where various scenarios/ environments are stored. scenario code consists of several functions:
    1) `make_world()`: creates all of the entities that inhabit the world (landmarks, agents, etc.), assigns their capabilities (whether they can communicate, or move, or both).
     called once at the beginning of each training session
    2) `reset_world()`: resets the world by assigning properties (position, color, etc.) to all entities in the world
    called before every episode (including after make_world() before the first episode)
    3) `reward()`: defines the reward function for a given agent
    4) `observation()`: defines the observation space of a given agent
    5) (optional) `benchmark_data()`: provides diagnostic data for policies trained on the environment (e.g. evaluation metrics)

You can create new scenarios by implementing the first 4 functions above (`make_world()`, `reset_world()`, `reward()`, and `observation()`).

# Miscellaneous

- `./multiagent/core.py`: contains classes for various objects (Entities, Landmarks, Agents, etc.) that are used throughout the code.

- `./multiagent/rendering.py`: used for displaying agent behaviors on the screen.

# Execution:

1. bin/script.py loads - acts as main script
2. Loads scenario
    - Uses scenario to generate world
3. Loads mutli-agent enviroment given scenario settings and world
4. Renders environment (initial render)
5. Assigns policies (algorithms) for each agent
    - stored as policies[] list
6. Resets environment
7. Infinite while loop
    1. Makes a list of actions, one action per policy
    2. Performs one environment step using entire action list
    3. Re-render
