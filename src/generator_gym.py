import os
import yaml
import time
import numpy as np
import gym
import pickle
from src.agents import RandomAgent, DQNAgent


def main():
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    run(cfg)


def run(cfg: dict):
    print('Loading environment {}.'.format(cfg['GYM_ENV']))
    env = gym.make(cfg['GYM_ENV'])
    env.reset()
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    state = np.zeros((1, observation_space))

    print('Creating Agent.')
    agent = DQNAgent(observation_space, action_space)
    
    states = []
    actions = []
    rewards = []
    dones = []
    new_states = []

    print('Starting training with {} steps.'.format(cfg['STEPS']))
    mean_step_reward = []
    reward_cur_episode = []
    reward_last_episode = 0
    episode = 1
    start_time = time.time()
    for steps in range(1, cfg['STEPS'] + 1):

        action = agent.act(state)
        env.render()
        new_state, reward, done, info = env.step(int(action))

        new_state = np.reshape(new_state, (1, observation_space))

        agent.add_experience(state, action, reward, done, new_state)
        agent.learn()

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        new_states.append(new_state)

        mean_step_reward.append(reward)
        reward_cur_episode.append(reward)

        if steps % cfg['VERBOSE_STEPS'] == 0:
            elapsed_time = time.time() - start_time
            print('Ep. {0:>4} with {1:>7} steps total; {2:8.2f} last ep. reward; {3:+.3f} step reward; {4}h elapsed' \
                  .format(episode, steps, reward_last_episode, np.mean(mean_step_reward), format_timedelta(elapsed_time)))
            mean_step_reward = []

        if done:
            reward_last_episode = sum(reward_cur_episode)
            reward_cur_episode = []
            episode += 1
            env.reset()

        state = new_state

    print('Closing environment.')
    env.close()

    print('Saving generated experiences.')
    states = np.vstack(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)
    new_states = np.vstack(new_states)
    with open('data/' + cfg['GYM_ENV'] + '.pkl', 'wb') as output:
        data = (states, actions, rewards, dones, new_states)
        pickle.dump(data, output)


def format_timedelta(timedelta):
    total_seconds = int(timedelta)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)


if __name__ == '__main__':
    main()