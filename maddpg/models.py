import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 64)
        self.FC2 = nn.Linear(64, 64)
        self.FC3 = nn.Linear(64, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.softmax(self.FC3(result))
        return result


class Critic(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 64)       # nn.Linear(obs_dim+act_dim, 64)
        self.FC2 = nn.Linear(64+dim_action, 64)
        self.FC3 = nn.Linear(64, 1)

    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 1)    # concatenate tensors in columns
        result = F.relu(self.FC2(combined))
        result = self.FC3(result)
        return result








