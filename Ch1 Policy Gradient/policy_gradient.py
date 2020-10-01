from itertools import count
import torch
import torch.nn as nn 
import numpy as np
import gym 


# create environment
env = gym.make('CartPole-v0')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n

# create policy
class Policy(nn.Module):
    def __init__(self, state_space, action_space):
        super(Policy).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.linear1 = nn.Linear(state_space, 10)
        self.linear2 = nn.Linear(10, action_space)

    def forward(x)