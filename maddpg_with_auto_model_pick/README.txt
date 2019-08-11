To use the code, first clone the repository from openai/maddpg.
Then put train_with_model_pick.py under the path maddpg/experiments/
and put tf_util_new.py under the path maddpg/maddpg/common/

The train_with_model_pick.py appends a new option "--load-best", which 
loads the best model (the model which produces the highest mean episode reward).
Use it with "--display" command.

While training, the program will automatically stores the best model in 
<filename>_best directory. The model will be stored at /tmp/policy_best 
on default.