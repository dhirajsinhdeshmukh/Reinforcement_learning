"""

### NOTICE ###
DO NOT revise this file

"""

import argparse
import numpy as np
from environment import Environment
	
import time
 
# Wait for 5 seconds
time.sleep(5)
seed = 11037
# seed = 512

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
        print(i)
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0
        counter = 0
        #playing one game
        while(not done):
            env.render()
            time.sleep(0.005)
            action = agent.make_action(state,counter, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            counter +=1
        rewards.append(episode_reward)


    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
