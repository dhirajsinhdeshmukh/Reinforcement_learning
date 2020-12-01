#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np

# import gc


class DQN(nn.Module):
    """Initialize a deep Q-learning network
    
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    
    This is just a hint. You can build your own structure.
    """
    def __init__(self, n_actions):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # gc.enable()
        self.input_shape = [4,84,84]
        self.conv = nn.Sequential(
        nn.BatchNorm2d(self.input_shape[0]),    
        nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
        # nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Flatten()
        )
        conv_out_size = self._get_conv_out(self.input_shape)
        self.fc = nn.Sequential(
        nn.Linear(conv_out_size, 512),
        nn.ReLU(),
        nn.Linear(512, n_actions),
        # nn.ReLU()
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        # print(o.shape)
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        x = torch.Tensor(x).to('cuda')
        conv_out = self.conv(x)
        return self.fc(conv_out)
        
        ###########################




class DuelingDQN(nn.Module):
    def __init__(self, n_actions):
        super(DuelingDQN, self).__init__()
        
        # gc.enable()

        self.input_shape = [4,84,84]

        self.conv = nn.Sequential(   
        nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
        # nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Flatten()
        )
        
        conv_out_size = self._get_conv_out(self.input_shape)

        self.advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        # print(o.shape)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = torch.Tensor(x).to('cuda')
        x = x/255
        x = self.conv(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()


class DuelingDQN_2(nn.Module):
    def __init__(self, n_actions):
        super(DuelingDQN_2, self).__init__()
        
        # gc.enable()

        self.input_shape = [4,84,84]

        self.conv = nn.Sequential(   
        nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
        # nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Flatten()
        )
        
        conv_out_size = self._get_conv_out(self.input_shape)

        self.advantage_1 = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        
        self.value_1 = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage_2 = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        
        self.value_2 = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage_3 = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        
        self.value_3 = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage_4 = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        
        self.value_4 = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        # print(o.shape)
        return int(np.prod(o.size()))

    def forward(self, x,y):
        x = torch.Tensor(x).to('cuda')
        x = x/255
        x = self.conv(x)
        if(y==1):
            advantage = self.advantage_1(x)
            value     = self.value_1(x)
        elif(y==2):
            advantage = self.advantage_2(x)
            value     = self.value_2(x)
        elif(y==3):
            advantage = self.advantage_3(x)
            value     = self.value_3(x)
        elif(y==4):
            advantage = self.advantage_4(x)
            value     = self.value_4(x)


        return value + advantage  - advantage.mean()

    def play(self, x):
        x = torch.Tensor(x).to('cuda')
        x = x/255
        x = self.conv(x)
        
        advantage_1 = self.advantage_1(x)
        value_1     = self.value_1(x)
        one = value_1 + advantage_1  - advantage_1.mean()
        
        advantage_2 = self.advantage_2(x)
        value_2     = self.value_2(x)
        two = value_2 + advantage_2  - advantage_2.mean()
        
        advantage_3 = self.advantage_3(x)
        value_3     = self.value_3(x)
        three = value_3 + advantage_3  - advantage_3.mean()
        
        advantage_4 = self.advantage_4(x)
        value_4     = self.value_4(x)
        four = value_4 + advantage_4  - advantage_4.mean()


        return (one + two + three + four)/4