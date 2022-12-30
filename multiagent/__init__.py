from gym.envs.registration import register
import warnings

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

warnings.warn("This code base is no longer maintained, and is not expected to be maintained again in the future. \n' \
'For the past handful of years, these environments been maintained inside of PettingZoo (see https://pettingzoo.farama.org/environments/mpe/ \n).' \
'This maintained version includes documentation, support for the PettingZoo API, support for current versions of Python, numerous bug fixes, \n'' \
'support for installation via pip, and numerous other large quality of life improvements. We encourage researchers to consider switching to this \n' \
'version for any purposes other than comparing to results run on this version of the environments.'
