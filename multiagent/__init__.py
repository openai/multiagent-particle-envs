import os
import warnings

from gym.envs.registration import register

# Multiagent envs
# ----------------------------------------

register(
    id='MultiagentSimple-v0',
    entry_point='multiagent.envs:SimpleEnv',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)

register(
    id='MultiagentSimpleSpeakerListener-v0',
    entry_point='multiagent.envs:SimpleSpeakerListenerEnv',
    max_episode_steps=100,
)

warnings.warn("This code base is no longer maintained, and is not expected to be maintained again in the future. \n"
              "For the past handful of years, these environments been maintained inside of PettingZoo (see "
              "https://pettingzoo.farama.org/environments/mpe/). \nThis maintained version includes documentation, "
              "support for the PettingZoo API, support for current versions of Python, numerous bug fixes, \n"
              "support for installation via pip, and numerous other large quality of life improvements. \nWe "
              "encourage researchers to switch to this maintained version for all purposes other than comparing "
              "to results run on this version of the environments. \n")

if os.getenv('SUPPRESS_MA_PROMPT') != '1':
    input("Please read the raised warning, then press Enter to continue... (to suppress this prompt, please set the environment variable `SUPPRESS_MA_PROMPT=1`)\n")
