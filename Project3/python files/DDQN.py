#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import sys
# import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from multipledispatch import dispatch 
from torch.utils.tensorboard import SummaryWriter
from agent import Agent
from dqn_model import DuelingDQN




import gc
"""
you can import any package and define any extra function as you need
"""

seed = 11037
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class ReplayBuffer(object):
    
    def __init__(self):

        gc.enable()

        self.buffer_size = 100000
        self.buffer_train = 50000
        self.buffer = []

        self.batch_size = 32

    def add_to_buffer(self,counter,buffer_sample):
        if(counter < self.buffer_size):
            self.buffer.append(buffer_sample)
        else:
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

            del state_buffer
            del next_state_buffer
            del action_buffer
            del reward_buffer
            del done_buffer
        

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

        gc.enable()

        self.seed = 11037
        random.seed(self.seed)
        np.random.seed(self.seed)
        T.manual_seed(self.seed)
        env.seed(self.seed)

        self.env=env
        self.env2=env
        self.gama = 0.99
        self.learning_rate = 1e-5
        self.traning_stop = 8000.0
        self.output_shape = env.action_space.n 
        self.train_frequency = 4
        self.clipping = 1
        self.device=T.device('cuda')

        self.epsilon = 0.00001
        self.epsilon_stop = 0.000001
        self.decay_cycles = 1000000
        self.epsilon_slope = (self.epsilon - self.epsilon_stop)/self.decay_cycles

        self.counter = 0
        self.best_reward = 0.0
        self.mean_reward = []
        self.episode_reward = 0.0
        self.game_reward = 0.0

        self.replay_buffer = ReplayBuffer() 

        self.net = DuelingDQN(self.output_shape).to(self.device)
        self.target_net = DuelingDQN(self.output_shape).to(self.device)
        self.load_network()
        self.target_net.load_state_dict(self.net.state_dict())
        self.sync_traget_net = 3000
        # print(self.net)
        # print(self.target_net)
        
        self.loss = nn.MSELoss()
        self.optimizer_net = optim.Adam(self.net.parameters(), lr=self.learning_rate )
        
        self.writer = SummaryWriter(comment = 'DDQN')

        self.sync = 0

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.net.load_state_dict(T.load("next6.bat"))


    def sync_target_network(self):
        self.target_net.load_state_dict(self.net.state_dict())
        print("target_network sync",end = "\r")

    def save_network(self):
        T.save(self.net.state_dict(), "next6.bat")    
        print("target_network updated",end = "\r")

    def load_network(self):
        self.net.load_state_dict(T.load("next3.bat"))
        print("network sync",end = "\r")

        # net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))


    def init_game_setting(self):
        pass


    def make_action(self, state, test=True):
        q_val = self.net.forward(state.reshape((1,)+state.shape)).detach()#.cpu().numpy()
        action = int(T.max(q_val, dim=1)[1])
        # min_val = abs(np.min(q_val))
        # q_val = q_val + min_val
        # q_val = q_val / np.sum(q_val)
        # q_val = np.squeeze(q_val)
        # action = np.random.choice(np.arange(4), p = q_val)

        # action = int(T.max(q_val, dim=1)[1])
        # if random.random() < 0.01:
        #     action = self.env.action_space.sample()
        # else:
        #     q_val = self.net.forward(state.reshape((1,)+state.shape)).detach()
        #     action = int(T.max(q_val, dim=1)[1])
        return action


    def choose_action(self,state):
        if(len(self.mean_reward) % 15< 2):
            epsilon = 0.1
        elif(len(self.mean_reward) % 15 < 4):
            epsilon = 0.01
        elif(len(self.mean_reward) % 15 < 6):
            epsilon = 0.001
        else: 
            epsilon = self.epsilon

        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            q_val = self.net.forward(state.reshape((1,)+state.shape)).detach()
            action = int(T.max(q_val, dim=1)[1])
        return action

    def train_model_step(self,rewards_tensor,dones_tensor,action_tensor,q_values_net,next_q_values_net,next_q_values_target):
        # gc.enable()
        with T.no_grad():
            Q_actions =  T.max(next_q_values_net, dim=1)[1].unsqueeze(1)
            Q_max = next_q_values_target.gather(dim = 1, index = Q_actions).squeeze()
            Q_target = rewards_tensor + self.gama * Q_max * (1 - dones_tensor)
                    
        Q_val = q_values_net.gather( dim=1, index = action_tensor).squeeze()
                    
        loss = self.loss(Q_val, Q_target)
        self.net.zero_grad()
        self.target_net.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.net.parameters()), self.clipping)
        self.optimizer_net.step()

    def test(self,env, total_episodes=100):
        rewards = []
        env.seed(seed)
        for i in range(total_episodes):
            state = env.reset()
            self.init_game_setting()
            done = False
            episode_reward = 0.0
            print(i, end = "\r")
            #playing one game
            while(not done):
                # env.render()
                action = self.make_action(state, test=True)
                state, reward, done, info = env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
        print('Run %d episodes'%(total_episodes),end = "\r")
        print('Mean:', np.mean(rewards),end = "\r")

        return np.mean(rewards)

    def train(self):

        state = self.env.reset()

        while(self.best_reward < self.traning_stop):
            
            # gc.enable()
            # if(self.counter % 1000000 == 0):
            #     self.epsilon = 0.1
            #     self.learning_rate = self.learning_rate/2
            #     self.optimizer_net = optim.Adam(self.net.parameters(), lr=self.learning_rate )

            action = self.choose_action(state) 

            next_state , reward , done , info  = self.env.step(action)
            buffer_sample = (state, action, int(reward - done*10), next_state, done)
            self.replay_buffer.add_to_buffer(self.counter,buffer_sample)
            self.episode_reward += int(reward - done*10)

            state = None
            state = next_state

            if(self.counter > self.replay_buffer.buffer_train):
                self.epsilon = max(self.epsilon-self.epsilon_slope,self.epsilon_stop)
                if(self.counter % self.train_frequency == 0):
                    states,actions,rewards,next_states,dones = self.replay_buffer.sample_batch(self.counter)
                    
                    rewards_tensor = T.Tensor(rewards).to(self.device)
                    dones_tensor =  T.Tensor(dones).to(self.device)
                    action_tensor = T.LongTensor(actions).view(-1,1).to(self.device)
                    q_values_net      = self.net.forward(states)
                    next_q_values_net = self.net.forward(next_states).detach()
                    next_q_values_target = self.target_net.forward(next_states).detach() 

                    self.train_model_step(rewards_tensor,dones_tensor,action_tensor,q_values_net,next_q_values_net,next_q_values_target)
                       
                    if(self.counter % self.sync_traget_net == 0):
                        self.sync_target_network() 
                        self.sync = 1
                        # self.save_network()   
                        # print("Best mean reward updated %.3f , model saved" % (mean_reward))
                        # self.best_reward = mean_reward

                    del states
                    del next_states
                    del actions
                    del rewards
                    del dones
                    
            if (done):
                state=None
                state = self.env.reset()
                # self.game_reward += self.episode_reward
                # self.episode_reward = 0.0
                mean_reward = 0.0
                if info['ale.lives'] == 0:
                    self.game_reward = self.episode_reward
                    self.episode_reward = 0.0
                    self.mean_reward.append(self.game_reward)
                    mean_reward = np.mean([self.mean_reward[-10:]])
                    self.game_reward = 0.0
                    print("%d:Traning_iterations %d games, mean reward %.3f, epsilon %.2f"% (self.counter - self.replay_buffer.buffer_train, len(self.mean_reward), mean_reward, 
                        self.epsilon),end = "\r")
                   

                    self.writer.add_scalar("epsilon", self.epsilon, self.counter)
                    self.writer.add_scalar("mean_reward", mean_reward, self.counter)
                    self.writer.add_scalar("reward", self.game_reward, self.counter)
                
                if (self.best_reward < mean_reward ):
                        self.save_network()   
                        print("Best mean reward updated %.3f , model saved" % (mean_reward),end = "\r")
                        self.best_reward = mean_reward

            # print(self.counter)
            self.counter += 1

        self.env.close()
        self.writer.close()
                        



if __name__ == '__main__':
    pass