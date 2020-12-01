#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import collections 
import os
import sys
import matplotlib.pyplot as plt


import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from multipledispatch import dispatch 
from torch.utils.tensorboard import SummaryWriter
from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

seed = 2

T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class ReplayBuffer(object):
    
    def __init__(self,buffer_sample):
        self.buffer_size =100000
        self.buffer_train = 50000
        self.buffer = [buffer_sample]*self.buffer_size

        self.batch_size = 32

    def add_to_buffer(self,counter,buffer_sample):
        index_buffer = counter % self.buffer_size
        self.buffer[index_buffer] = buffer_sample

    def sample_batch(self, counter):
        len_buffer = min(counter,self.buffer_size)
        index_array = np.random.choice(len_buffer,self.batch_size,replace=False)

        # print(index_array)
    
        states, actions, rewards, next_states, dones = [],[],[],[],[]
        for i in index_array:
            
            state_buffer, action_buffer, reward_buffer, next_state_buffer, done_buffer = self.buffer[i]
            states.append(state_buffer)
            actions.append(action_buffer)
            rewards.append(reward_buffer)
            next_states.append(next_state_buffer)
            dones.append(done_buffer)
        

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)



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

        self.seed = 2
        random.seed(self.seed)
        np.random.seed(self.seed)
        T.manual_seed(self.seed)
        env.seed(self.seed)

        self.env=env
        self.gama = 0.99
        self.learning_rate = 1e-5
        self.traning_stop = 800.0
        self.output_shape = env.action_space.n 
        self.train_frequency = 4
        self.clipping = 0.5
        self.device=T.device('cuda')

        self.epsilon = 0.1
        self.epsilon_stop = 0.01
        self.decay_cycles = 1000000
        self.epsilon_slope = (self.epsilon - self.epsilon_stop)/self.decay_cycles

        self.counter = 0
        self.best_reward = 0.0
        self.mean_reward = []
        self.episode_reward = 0.0
        self.game_reward = 0.0

        self.state = self.env.reset()
        self.action = 1 
        self.next_state , self.reward , self.done , self.info  = self.env.step(self.action)
        self.buffer_sample = (self.state, self.action, self.reward, self.next_state, self.done)

        self.replay_buffer = ReplayBuffer(self.buffer_sample) 

        self.net = DQN(self.output_shape).to(self.device)
        self.target_net = DQN(self.output_shape).to(self.device)
        self.load_network()
        self.target_net.load_state_dict(self.net.state_dict())
        self.sync_traget_net = 5000
        # print(self.net)
        # print(self.target_net)

        self.optimizer_net = optim.Adam(self.net.parameters(), lr=self.learning_rate )
        self.loss = nn.MSELoss()

        self.writer = SummaryWriter(comment = 'DQN_4')

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.net.load_state_dict(T.load("DQN_wednesday_13_from_12.bat"))


    def sync_target_network(self):
        self.target_net.load_state_dict(self.net.state_dict())
        print("target_network sync")

    
    def save_network(self):
        T.save(self.net.state_dict(), "DQN_wednesday_13_from_12.bat")    
        print("target_network updated")

    def load_network(self):
        self.net.load_state_dict(T.load("DQN_wednesday_12_from_11.bat"))
        print("network sync")

        # net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))


    def init_game_setting(self):
        pass


    def make_action(self, state, test=True):
        if random.random() < 1e-2:
            action = self.env.action_space.sample()
        else:
            q_val = self.net.forward(state.reshape((1,)+state.shape))
            action = T.argmax(q_val, dim=1).tolist()[0]
        return action


    def choose_action(self,state):
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q_val = self.net.forward(state.reshape((1,)+state.shape))
            action = T.argmax(q_val, dim=1).tolist()[0]
        return action

    def train(self):

        self.state = self.env.reset()

        while(self.best_reward < self.traning_stop):
            
            if(self.counter % 1000000 == 0):
                self.epsilon = 0.1
                self.learning_rate = self.learning_rate/2
                self.optimizer_net = optim.Adam(self.net.parameters(), lr=self.learning_rate )

            self.action = self.choose_action(self.state) 

            self.next_state , self.reward , self.done , self.info  = self.env.step(self.action)
            self.buffer_sample = (self.state, self.action, self.reward, self.next_state, self.done)
            self.replay_buffer.add_to_buffer(self.counter,self.buffer_sample)
            self.episode_reward += self.reward

            self.state = self.next_state


            if(self.counter > self.replay_buffer.buffer_train):
                self.epsilon = max(self.epsilon-self.epsilon_slope,self.epsilon_stop)
                if(self.counter % self.train_frequency == 0):
                    states,actions,rewards,next_states,dones = self.replay_buffer.sample_batch(self.counter)
                    
                    rewards_tensor = T.Tensor(rewards).to(self.device)
                    dones_tensor =  T.Tensor(dones).to(self.device)

                    with T.no_grad():
                        Q_max = T.max(self.target_net.forward(next_states), dim=1)[0]
                        Q_target = rewards_tensor + self.gama * Q_max * (1 - dones_tensor)
                    
                    Q_val = self.net.forward(states).gather(1, T.LongTensor(actions).view(-1,1).to(self.device)).squeeze()
                    
                    loss = self.loss(Q_val, Q_target)
                    self.optimizer_net.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(list(self.net.parameters()), self.clipping)
                    self.optimizer_net.step()

                    if(self.counter % self.sync_traget_net == 0):
                        self.sync_target_network()
                    
                    del states
                    del next_states
                    del actions
                    del rewards
                    del dones


                    
            if (self.done):
                self.state = self.env.reset()
                self.game_reward += self.episode_reward
                self.episode_reward = 0.0
                mean_reward = 0.0
                if self.info['ale.lives'] == 0:
                    self.mean_reward.append(self.game_reward)
                    mean_reward = np.mean([self.mean_reward[-100:]])
                    self.game_reward = 0.0
                    print("%d:Traning_iterations %d games, mean reward %.3f, epsilon %.2f" % (self.counter - self.replay_buffer.buffer_train, len(self.mean_reward), mean_reward, 
                        self.epsilon))

                    self.writer.add_scalar("epsilon", self.epsilon, self.counter)
                    self.writer.add_scalar("mean_reward", mean_reward, self.counter)
                    self.writer.add_scalar("reward", self.game_reward, self.counter)

                if (self.best_reward < mean_reward):
                        self.save_network()   
                        print("Best mean reward updated %.3f , model saved" % (mean_reward))
                        self.best_reward = mean_reward

            # print(self.counter)
            self.counter += 1


        self.env.close()
        self.writer.close()
                        



if __name__ == '__main__':
    pass