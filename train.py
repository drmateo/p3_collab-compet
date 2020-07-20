# MIT License
#
# Copyright (c) 2020 Carlos M. Mateo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
from unityagents import UnityEnvironment
import numpy as np

import torch
import argparse
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from itertools import count

from ddpg_agent import Agent, Config

def train_ddpg(agent, env, brain_name, num_agents, n_episodes=1000, max_t=2000, print_every=100, early_stop=True):
    max_score=0.5
    scores_deque = deque(maxlen=print_every)
    scores_hist = []
    scores_avg_hist = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)        
        agent.reset()
        
        for i_step in range(max_t):
            actions = agent.act(states)                        # select an action (for each agent)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agent.step(states, actions, rewards, next_states, dones)
            
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step

            if np.any(dones):
                break
    
        score = scores.max()
        scores_deque.append(score)
        scores_hist.append(score)
        scores_avg_hist.append(np.mean(scores_deque))
        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}\tMax Score: {:.2f}\t\tMin Score: {:.2f}'.format(
            i_episode, score, np.mean(scores_deque), np.max(scores_deque), np.min(scores_deque)), end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}\tMax Score: {:.2f}\t\tMin Score: {:.2f}'.format(
                i_episode, score, np.mean(scores_deque), np.max(scores_deque), np.min(scores_deque)))

        if np.mean(scores_deque) >= max_score:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_{:05d}_{:03d}.pth'.format(i_episode, int(max_score*100)))
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_{:05d}_{:03d}.pth'.format(i_episode, int(max_score*100)))
            max_score=np.mean(scores_deque)
            if early_stop:
                break

    return scores_hist, scores_avg_hist


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('DDPG train')
    parser.add_argument('--num_eps', required=False, type=int, default=10000, help='number of episodes (default 10000)')
    parser.add_argument('--early_stop', required=False, type=bool, default=False, help='Stop training once agent reaches 0.5 points (averaged)')
    args = parser.parse_args()

    env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # size of each state 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

    # Agent configuration
    config = Config(num_agents)
    config.buffer_size = int(1e5)
    config.batch_size = 512

    config.tau = 1e-3
            
    config.actor_hidden_drop = 0.4
    config.actor_hidden_units = (128, 128)
    config.lr_actor = 5e-4

    config.critic_hidden_drop = 0.4
    config.critic_hidden_units = (128-action_size, 128)
    config.lr_critic = 5e-4

    agent = Agent(state_size, action_size, config=config, random_seed=1234)
    print('Model:')
    print(agent.actor_local)
    print(agent.critic_local)

    scores, scores_avg_hist = train_ddpg(agent, env, brain_name=brain_name, num_agents=num_agents, n_episodes=args.num_eps, early_stop=args.early_stop)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores, 'b-')
    plt.plot(np.arange(1, len(scores)+1), scores_avg_hist, 'r-')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    print('')

    env.close()
