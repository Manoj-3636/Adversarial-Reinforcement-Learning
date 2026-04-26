from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from environment.config import NN_ARCH

def xavier_init(module):
    if isinstance(module,nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias,0)

class Actor(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(state_dim,NN_ARCH["actor_hidden_units"])
        self.fc2 = nn.Linear(NN_ARCH["actor_hidden_units"],action_dim)

        self.apply(xavier_init)

    def forward(self,state):
        x = torch.tanh(self.fc1(state))
        return torch.sigmoid(self.fc2(x))
    
def he_normal_init(module):
    if isinstance(module,nn.Linear):
        nn.init.kaiming_normal_(module.weight,mode='fan_in',nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias,0)

class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim,NN_ARCH["critic_hidden_units"])
        self.fc2 = nn.Linear(NN_ARCH["critic_hidden_units"],1)

        self.apply(he_normal_init)

    def forward(self,state,action):
        x = torch.cat([state,action],1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

@dataclass
class Policy:
    model:dict
    itr:int