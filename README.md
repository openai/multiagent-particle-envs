# Multi-Agent Goal-Driven Communication Environment

Environment implementation for the [Goal-Driven Communication](https://openai.quip.com/zMfIAOFifa8o) project.
Consists of a simple particle world with a continuous observation and discrete action space, along with some basic simulated physics.

## Getting started:

- To install, `cd` into the root directory and type `pip install -e .`

- To interactively view moving to landmark scenario (see others in ./scenarios/):
`bin/interactive.py --scenario simple.py`

- Known dependencies: OpenAI gym, numpy, tensorflow

- Code to train and evaluate models on environments presented here can be found at https://github.com/openai/multiagent-rl

## Code structure

- `environment.py`: contains code for environment simulation (interaction physics, _step() function, etc.)

- `core.py`: contains classes for various objects (Entities, Landmarks, Agents, etc.) that are used throughout the code

- `/scenarios/`: folder where various scenarios/ environments are stored. scenario code consists of several functions:
    1) `make_world()`: creates all of the entities that inhabit the world (landmarks, agents, etc.), assigns their capabilities (whether they can communicate, or move, or both).
     called once at the beginning of each training session
    2) `reset_world()`: resets the world by assigning properties (position, color, etc.) to all entities in the world
    called before every episode (including after make_world() before the first episode)
    3) `reward()`: defines the reward function for a given agent
    4) `observation()`: defines the observation space of a given agent
    5) (optional) `benchmark_data()`: provides diagnostic data for policies trained on the environment (e.g. evaluation metrics)

You can create new scenarios by implementing the first 4 functions above.

- A list of currently available scenarios (and short descriptions) can be found here:
https://docs.google.com/document/d/1Z5LnFiL0rt-qGUdmNBt4m0SrAuNdBS8O-F5i17sQjVY/edit
