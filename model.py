import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units=(512,256), drop_p=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_units (list): Number of nodes in hidden layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dims = (state_size, ) + hidden_units

        self.normalizer = nn.BatchNorm1d(state_size)
        self.dropout = nn.Dropout(p=drop_p)

        self.fc_hidden = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(self.dims[:-1], self.dims[1:])])
        self.fc_out = nn.Linear(self.dims[-1], action_size)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fc_hidden:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.normalizer(state)
        for i, layer in enumerate(self.fc_hidden):
            x = F.relu(layer(x))
            if i > 0: x = self.dropout(x)

        return F.tanh(self.fc_out(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units=(512,256), drop_p=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_units (list): Number of nodes in hidden layers
            drop_p (float): Proba to dropout nodes in hidden layers (default: 0.2)
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dims = (state_size, ) + hidden_units

        self.normalizer = nn.BatchNorm1d(state_size)
        self.dropout = nn.Dropout(p=drop_p)

        self.fc_hidden = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            if i == 1:
                self.fc_hidden.append(nn.Linear(dim_in+action_size, dim_out))
            else:
                self.fc_hidden.append(nn.Linear(dim_in, dim_out))
        self.fc_out = nn.Linear(self.dims[-1], 1)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fc_hidden:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.normalizer(state)
        for i, layer in enumerate(self.fc_hidden):
            if i == 1: x = torch.cat((x, action), dim=1)
            x = F.relu(layer(x))
            if i > 0: x = self.dropout(x)

        return self.fc_out(x)
