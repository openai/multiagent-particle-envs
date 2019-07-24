# Idea table:

Generated at: https://www.tablesgenerator.com/markdown_tables

|      | Possible Actions        | Rewards per Outcome                                                                                                        | Properties of other entities          | Nash Equilibrium          | Other Notes   |
| ---- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------- | ------------- |
| #1   | Expand, attack, trade   | Expanding + attacking spends resources for greater resource bonuses later.                                                 | No other entities other than agents   | Attack                    |               |
|      |                         | Trading gives bonus resources for both agents                                                                              |                                       |                           |               |
| #2   | Move x steps            | Reward = progress in last step                                                                                             | No other entities other than agents   | Move as far as possible   |               |
| #3   | move, attack, loot, rest |  Staying alive, get gear, kills, winning                                                                                                         |     No other entities other than agents                                 |     live and loot                      |              |
| #4   |                         |                                                                                                                            |                                       |                           |               |
| #5   |                         |                                                                                                                            |                                       |                           |               |

# Details:
## Idea 1. (Risk but on a grid)
Grid based cell game, each agent starts with 1 cell on some part of the grid. Agents use resources to expand, attack, or trade with neighboring cells. Every turn agents gain a set amount of resources based on area of agent's cells. For every neighboring cell, if it is not occupied, the agent can choose to spend resources to expand into the area, or not. If the cell is occupied, the agent can choose to attack, or trade. Attacking allows for the takeover of the cell and requires the agent to spend resources. Trading requires the agent to give resources to the other agent, but if both agents decide to trade, they can recieve some bonus based on who gave more resources. If one agent attacks, and the other trades, the attacker automatically wins. If both attack, the agent that spent more resources to attack wins. Resource costs and bonuses can be tweaked to ensure fairness and balance.
### Examples: 
Agent A and Agent B are neighbors: if A trades 2 resources, and B trades 4 resources, A could gain 4(from B) + 2(bonus includes how much given) + 1(some multiplier of how much was given in this case 0.5 for giving less) resulting in net +5, B would gain 2(from A) + 4(given) + 4(multiplier bonus of 1 for giving more) resulting in +6 \
If A attacks B, spending 5 resources; B attempts to trade 4 resources, A takes over some area of B and gains 4 resources from B's trade with net gain of -1 resource and + some area; B has a net gain of -4 resources and -some area. \
If A attacks B, spending 5 resources; B attacks A spending 6 resources, B takes some area of A. A has a net gain of -5 resources and -some area; B has a net gain of -5 resources and -some area.

### Possible expansion:
Add defend action, which blocks attack, but opponent agent gains bigger bonus resource if they try to trade. 

## Idea 2. Race

#### World
- 2D plane where agents try to race to their landmark.
- Agents only move forward/backward parallel to each other
- All agents start from the same location

#### Rules
- Agents can take any x number of steps to advance to the landmark.
- If the sum of all the steps taken by the agents(y) exceeds z, then all agents that moved get moved backwards w steps. 
- Agents are rewarded based off of how many steps they are successfully able to take per turn.
    - Say each agent tries to take 10 steps, but because too many agents are trying to move in a turn, they all get moved back 5 steps. In this case, their reward would be -5, even though they tried to take 10 steps.

#### Variables
- Parameters y, z, and the initial distance for each agent to the landmark can be varied for balance and to compare agent behavior.

#### Notes
- The landmarks don't actually get taken into account for the rewards or observation, it's simply aiding visualization of how much progess each agent is able to make.
- Agents will either have to be moved by the scenario via physics, or they can move based off of the reward recieved on their next action, the following turn.

## Idea 3. Hunger Games

#### World
- 10 x 10 plane where agents try to be the last survivor
- 12 agents start equidistant from each other in a circle
    - Middle of circle is high tier loot
- Set loot spawns with a set tier, but random loot
- Structures agents can enter and be hidden from sight

#### Agent
- Main Attributes
    - Attack Range
    - Attack Power
    - Def
    - HP
    - Stamina
- Choices:
    - Loot
        - Sight limited
    - Attack an adjacent agent
        - Additional attack options possible with certain loot
        - Uses Stamina
    - Move
        - Walk one space
        - Run 2x fast w/ Stamina
    - Rest
        - Recover HP/Stamina
#### Rewards
- Kills are not intrinsically rewarded
- Looting from chests/bodies result in a set reward value per tier/killcount, and additional reward from net stat gain
- Time alive gives slight reward with each tick
- Winning gives the highest reward

#### Agent Variables
- Environment Knowledge
    - Excludes:
        Chest loot status
        Chest loot items
        Alive/Dead Enemy location
- Sight
- Self position
- Attributes
- Loot
- Kill count
- Kill counts of other agents
- List of Alive Agents
- List of Dead Agents
- Attributes of agents in Sight
    
#### Notes
- Co-op can be implemented where agents spawn with a teammate they cannot attack, and exchange loot with.
- Combat can function similarly to D&D involving some RNG

