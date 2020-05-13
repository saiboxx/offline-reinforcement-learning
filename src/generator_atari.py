import os
import yaml
import time
import numpy as np
import gym
import pickle
from src.agents import RandomAgent, DQNAgent
from src.utils.data import DataSaver, Summary


def main():
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    run(cfg)


def run(cfg: dict):
    print('Loading environment {}.'.format(cfg['GYM_ENV']))
    env = gym.make(cfg['GYM_ENV'])
    env.reset()
    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    state = np.zeros(observation_space)

    print('Creating Agent.')
    agent = DQNAgent(observation_space, action_space)
    saver = DataSaver(cfg['DATA_PATH'])
    summary = Summary(cfg['SUMMARY_PATH'], agent.name)
    agent.add_summary_writer(summary)

    print('Starting training with {} steps.'.format(cfg['STEPS']))
    mean_step_reward = []
    reward_cur_episode = []
    reward_last_episode = 0
    episode = 1
    start_time = time.time()
    start_time_episode = time.time()

    for steps in range(1, cfg['STEPS'] + 1):

        action = agent.act(state)
        env.render()
        new_state, reward, done, info = env.step(int(action))

        agent.add_experience(state, action, reward, done, new_state)
        agent.learn()
        saver.save(state, action, reward, done, new_state)

        mean_step_reward.append(reward)
        reward_cur_episode.append(reward)

        summary.add_scalar('Step Reward', reward)

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
            duration_last_episode = time.time() - start_time_episode
            start_time_episode = time.time()
            summary.add_scalar('Episode Reward', reward_last_episode, True)
            summary.add_scalar('Episode Duration', duration_last_episode, True)
            summary.adv_episode()
            summary.writer.flush()

        state = new_state
        summary.adv_step()

    print('Closing environment.')
    env.close()
    saver.close()


def format_timedelta(timedelta):
    total_seconds = int(timedelta)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)


if __name__ == '__main__':
    main()
