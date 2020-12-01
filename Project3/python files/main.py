import argparse
from test import test
from environment import Environment
import matplotlib.pyplot as plt
import numpy as np
import time

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn_model import DQN

import gc

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from test_main import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from test_main import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)

  ### Check Environment output ### 
  ######################################################## 
    # env_name = args.env_name or 'BreakoutNoFrameskip-v4'
    # env = Environment(env_name, args, atari_wrapper=True)
    # ob = env.reset() 
    # for i in range(10): 
    #     ob,_,_,_ = env.step(0)
    #     ob = ob.T
    #     ob = ob.round(1)
    #     print(ob.shape)
    #     print(np.max(ob))
    #     plt.imshow(ob)
    #     plt.show()
  #########################################################
       


    ### test Network model ###
    ########################################

    # net = DQN(4)
    # obs = T.FloatTensor(ob).unsqueeze(0)
    # x = net.forward(obs)
    # print(net)
    # print(x)

    ########################################

    ### check make_action ###
    ########################################
    
    # from agent_dqn import Agent_DQN
    # agent = Agent_DQN(env, args)
    # agent.make_action(ob)
    ########################################




