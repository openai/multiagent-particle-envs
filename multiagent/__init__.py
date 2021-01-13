from gym.envs.registration import register

# Multiagent envs
# ----------------------------------------

register(
    id='MultiagentSimple-v0',
    entry_point='multiagent.envs:MultiAgentEnv',
)
