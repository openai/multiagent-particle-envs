To use a model, first rename the folder to "policy". Then move the folder to /tmp.
For the folder with name starting with "policy_best", rename it to "policy_best"
and move it to /tmp.

To use a model named policy, go to maddpg/experiments first. Run the following command:
python3 train.py --scenario <the name of the corresponding scenario> --display
                                     or
python3 train_with_model_pick.py --scenario <the name of the corresponding scenario> --display


To use a model named policy_best, go to maddpg/experiments first. Run the following command:
python3 train_with_model_pick.py --scenario <the name of the corresponding scenario> --display --load-best


Notes:
For simple_tagGT, there are parameters required to be set before running the scenario.

simple_tagGT_default: (emphasis on cooperation between predators)
	numPrey = 1    #Number of prey (runners)
    numPredators = 3    #Number of predators (catchers)
    preyIndRew = 1    #Negative Reward multiplier that prey get individually for being caught 
    preyGroupRew = 0    #Negative Reward multiplier that prey get for another prey being caught
    predIndRew = 1    #Reward multiplier that predators get individually for catching prey
    predGroupRew = 1    #Reward multiplier that predators get for another predator catching prey

simple_tagGT_prey_co-op: (emphasis on cooperation between prey)
	numPrey = 2    #Number of prey (runners)
    numPredators = 4    #Number of predators (catchers)
    preyIndRew = 1    #Negative Reward multiplier that prey get individually for being caught 
    preyGroupRew = 1    #Negative Reward multiplier that prey get for another prey being caught
    predIndRew = 1    #Reward multiplier that predators get individually for catching prey
    predGroupRew = 1    #Reward multiplier that predators get for another predator catching prey


Please uncomment the line 83-87(in reward function) when running the water_hole scenario.