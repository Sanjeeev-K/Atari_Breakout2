#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # # Number of Linear input connections depends on output of conv2d layers
        # # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
    
        # linear_input_size = convw * convh * 32
        # # print("Linear input size: ", linear_input_size)
        # self.head_prev = nn.Linear(linear_input_size, 256)
        # self.head_next = nn.Linear(256, num_actions)

        # **** START HERE ****************


        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        #self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #self.bn3 = nn.BatchNorm2d(64)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size = 5, stride = 2):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
    
        # linear_input_size = convw * convh * 32
        linear_input_size = 3136
        # print("Linear input size: ", linear_input_size)
        self.head_prev = nn.Linear(linear_input_size, 512)
        self.head_next = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # # print ('Neural network Input shape : ', x.size())
        # # print(x, type(x))
        # x = x.float()
        # # print("x input shape: ", x.shape)
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # # print("x_flatten?: ", (x.view(x.size(0), -1)).shape)
        # x = F.relu(self.head_prev(x.view(x.size(0), -1)))
        # # print("After Linear ", x.size())
        # x = self.head_next(x.view(x.size(0), -1))
        # # print("After Linear 2 ", x.size())
        # return x

        # RISHABH IMPLEMENTATION HERE #
        # print ('Neural network Input shape : ', x.size())
        # print(x, type(x))
        x = x.float()/255.0
        # print("x input shape: ", x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 3136)

        x = self.head_prev(x)
        # print("After Linear ", x.size())
        x = F.relu(x)
        x = self.head_next(x)
        # print("After Linear 2 ", x.size())
        return x
        ###########################
        # return x
