"""

### NOTICE ###
DO NOT revise this file

"""

import argparse
import numpy as np
from environment import Environment
import torch 
import time


seed = 11037

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=100):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0
        # print(i, end = "\r")
        #playing one game
        while(not done):
            env.render()
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from DDQN import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)




if __name__ == '__main__':
    args = parse()
    run(args)
    # from dqn_model import DQN
    # env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
    # model="DDQN.bat"
    # net = DuelingDQN(4).to('cuda')
    # net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    # # record_folder="video"  
    # visualize=True
    # for i in range(100):
    #     state = env.reset()
    #     total_reward = 0.0
    #     while True:
    #             # start_ts = time.time()
    #             if visualize:
    #                 env.render()
    #                 time.sleep(0.01)
    #             state_v =  state.reshape((1,)+state.shape)
    #             q_vals = net(state_v)
    #             values , indices = torch.max(q_vals,dim=1)
    #             action = int(indices)
                
    #             state, reward, done, _ = env.step(action)
    #             total_reward += reward
    #             if done:
    #                 break
    #             # if visualize:
    #             #     delta = 1/FPS - (time.time() - start_ts)
    #             #     if delta > 0:
    #             #         time.sleep(delta)
    #     print("Total reward: %.2f" % total_reward)
    #     env.close()

    # # if record_folder:
    # #         env.close()