To train:
=========
python train.py --scenario [scenario root name] --save-dir "./[scenario root name]/model"

To display:
===========
python train.py --scenario [scenario root name] --display --load-dir "[scenario root name]/model"


optional arguments:
====================

```
  -h, --help            show this help message and exit
  --scenario SCENARIO   name of the scenario script
  --max-episode-len MAX_EPISODE_LEN
                        maximum episode length
  --num-episodes NUM_EPISODES
                        number of episodes
  --num-adversaries NUM_ADVERSARIES
                        number of adversaries
  --good-policy GOOD_POLICY
                        policy for good agents
  --adv-policy ADV_POLICY
                        policy of adversaries
  --lr LR               learning rate for Adam optimizer
  --gamma GAMMA         discount factor
  --batch-size BATCH_SIZE
                        number of episodes to optimize at the same time
  --num-units NUM_UNITS
                        number of units in the mlp
  --exp-name EXP_NAME   name of the experiment
  --save-dir SAVE_DIR   directory in which training state and model should be
                        saved
  --save-rate SAVE_RATE
                        save model once every time this many episodes are
                        completed
  --load-dir LOAD_DIR   directory in which training state and model are loaded
  --restore
  --display
  --benchmark
  --benchmark-iters BENCHMARK_ITERS
                        number of iterations run for benchmarking
  --benchmark-dir BENCHMARK_DIR
                        directory where benchmark data is saved
  --plots-dir PLOTS_DIR
                        directory where plot data is saved
```
