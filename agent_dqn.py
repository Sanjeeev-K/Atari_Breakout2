#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN

##

from test import test

"""
you can import any package and define any extra function as you need
"""
import math
from environment import Environment as env
from itertools import count
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))



BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.00
EPS_END = 0.01
EPS_DECAY = 1e6
TARGET_UPDATE = 10000
LEARNING_RATE = 0.0001
REWARD_BUFFER_SIZE = 100
MEMORY_SIZE = 100000
NUM_EPISODES = 30000000
EPISODE_STEP_LIMIT = 10000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.memory = []
        self.env = env
        self.n_actions = env.env.action_space.n
        self.policy_net = DQN(4, self.n_actions).to(device).float()
        self.target_net = DQN(4, self.n_actions).to(device).float()
        # self.policy_net.load_state_dict(torch.load("best_weights_model.pt"))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.eps_threshold = EPS_START
        self.args = args
        self.test_count = 0
        self.max_reward = 0
        self.max_reward_so_far = 0
        self.reward_buffer = []
        self.flag = 0
        self.steps_done = 0
        # self.target_net.eval()

        self.test_mean_reward = 0

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = LEARNING_RATE)
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.policy_net.load_state_dict(torch.load("saved_model.pt"))
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # self.env.reset()
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        if not test:
            # print("Helllo")
            # global steps_done
            
            sample = random.random()

            self.eps_threshold = self.eps_threshold -  (EPS_START - EPS_END) / EPS_DECAY
            
            if self.eps_threshold < EPS_END:
                self.eps_threshold = EPS_END
            # print("Steps after increment ", self.steps_done)
            if sample > self.eps_threshold:
                with torch.no_grad():
                    
                    q_sa = self.policy_net( torch.from_numpy(observation).unsqueeze(0).to(device))
                    index = torch.argmax(q_sa.data,dim=1).item()

                    return index
            else:
                return np.random.randint(0, self.n_actions)
        else:            
            q_sa = self.policy_net( torch.from_numpy(observation).unsqueeze(0).to(device))
            index = torch.argmax(q_sa.data,dim=1).item()

            return index

        ###########################
        # return action
    
    def push(self, *args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.memory) >= 50000:
            self.memory.pop(0)
        self.memory.append(Transition(*args))

        if(len(self.memory)%500==0 or len(self.memory)>= 50000):
            print("Memory size : ", len(self.memory))
        ###########################
        
        
    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        transitions = random.sample(self.memory,BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        ###########################
        return batch
    
    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # reward_buffer = deque([]) 
        current_loss = 0.0
        mean_reward = 0.0
        for i_episode in range(NUM_EPISODES):
            # Initialize the environment and state
            # self.env.reset()
            # last_screen = get_screen()
            # current_screen = get_screen()
            state = self.env.reset()
            state = np.transpose(state,(2,0,1))
            # state = torch.tensor([state])
            episode_Reward = 0.0
            for t in range (EPISODE_STEP_LIMIT):
                # Render here
                # self.env.env.render()
                self.steps_done += 1

                action = self.make_action(state, False)
                # 'Transition',('state', 'action', 'next_state', 'reward', 'done'))
                    
                next_state, reward, done, _ = self.env.step(action)
                episode_Reward += reward
                

                next_state = np.transpose(next_state,(2,0,1))
                self.push(state, action, next_state, reward, done)

                # Move to the next state
                state = next_state
                
                # self.env.render()


                # Update the target network, copying all weights and biases in DQN
                # print("Steps : ",steps_done)
                if self.steps_done % TARGET_UPDATE == 0:
                    print("**********Updating Target********")
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # Perform one step of the optimization (on the target network)
                # optimize step start
                # print("Memory Size", len(self.memory))
                # print("Completed 10,000 steps")
                if len(self.memory) > 10000 and len(self.memory)%4 ==0:
                    if self.flag==0:
                        print("Crossed 10000")
                        self.flag = 1
                    batch = self.replay_buffer(BATCH_SIZE)
                    
                    # 'Transition',('state', 'action', 'next_state', 'reward', 'done'))
                    state_batch = torch.from_numpy(np.asarray(batch[0]))
                    action_batch = torch.from_numpy(np.asarray(batch[1]))
                    next_state_batch = torch.from_numpy(np.asarray(batch[2]))
                    reward_batch = torch.from_numpy(np.asarray(batch[3])).to(device)
                    done_batch = torch.from_numpy(np.asarray(batch[4])).to(device)

                    state_action_values = self.policy_net(state_batch.to(device)).gather(1,action_batch[:,None].to(device)).squeeze(1)
                    
                    q_max = self.target_net(next_state_batch.to(device)).max(1)[0].detach() * GAMMA
                    
                    q_max[done_batch] = 0

                    expected_state_action_values = (q_max) + reward_batch
                    #print (state_action_values.double().size())

                    #print (expected_state_action_values.double().size())
                    loss = F.smooth_l1_loss(state_action_values.double(), expected_state_action_values.double())
                   
                    current_loss = loss
                    # print("Episode : ", i_episode, ", iteration : ",t, " Loss :  ", current_loss, " Steps : ", steps_done," Epsilon : ", self.eps_threshold, " Mean Reward : ", mean_reward)

                    #optimze the model
                    self.optimizer.zero_grad()
                    loss.backward()
                 
                    self.optimizer.step()

                if done:
                    if len(self.reward_buffer)>= REWARD_BUFFER_SIZE:
                        self.reward_buffer.pop(0)
                    self.reward_buffer.append(episode_Reward)
                    mean_reward = np.mean(self.reward_buffer)
                    break
            
            
            if(i_episode%500 == 0):
                env2 = env('BreakoutNoFrameskip-v4', self.args, atari_wrapper=True, test=True)
                #test(self, env2, total_episodes=100)
                writer.add_scalar('Test Mean Reward', self.test_mean_reward, i_episode)
                if self.test_mean_reward > self.max_reward_so_far :
                    torch.save(self.policy_net.state_dict(), "best_weights_model.pt")
                    self.max_reward_so_far = self.test_mean_reward
                
        
            writer.add_scalar('Train Mean Reward', mean_reward, i_episode)
            writer.add_scalar('Training LOSS', current_loss, i_episode)

            
            # To calculate mean reward
            if i_episode % 100 == 0:
                mean_reward = sum(self.reward_buffer)/100
                # print("*****************")
                print("TRAIN Mean Reward after ", i_episode, " episodes is ", mean_reward, " Epsilon ", self.eps_threshold)
            if i_episode % 500 == 0:
                torch.save(self.policy_net.state_dict(), "saved_model.pt")
                print("Saved Model after ",i_episode, " episodes")
        self.env.env.close()
        self.writer.close()
        
        ###########################