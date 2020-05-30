import yaml
import time
import numpy as np
import gym
from tqdm import tqdm
import random
import torch
from torch import tensor
from src.atari.utils.networks import ConvEncoder
from src.atari.agents import RandomAgent, DQNAgent, DoubleDQNAgent
from src.atari.utils.data import DataSaver, Summary
from src.atari.utils.env_wrapper import wrap_env


def main():
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    run(cfg)


def run(cfg: dict):
    print('Loading environment {}.'.format(cfg['ATARI_ENV']))
    env = gym.make(cfg['ATARI_ENV'])
    env = wrap_env(env)
    env.reset()
    observation_space = env.observation_space.shape
    action_space = 3
    action_map = {0: 0, 1: 2, 2: 3}

    print('Creating Agent.')
    agent = DQNAgent(observation_space, action_space)
    saver = DataSaver(cfg['GEN_DATA_PATH'])
    summary = Summary(cfg['SUMMARY_PATH'], agent.name)
    agent.print_model()
    agent.add_summary_writer(summary)

    print('Initializing Encoder.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ConvEncoder()
    encoder.load_state_dict(torch.load(cfg['AUTO_SAVE_PATH'] + '/encoder.pt',
                                       map_location=device))
    encoder.to(device)
    encoder.eval()

    state = torch.zeros((1, 64)).to(device)
    state_mu = state_var = torch.zeros((4, 16)).to(device)

    print('Starting warm up.')
    for _ in tqdm(range(cfg['WARM_UP_STEPS'])):
        action = tensor(random.randint(0, action_space - 1)).to(device)
        if cfg['RENDER']:
            env.render()
        new_state, reward, done, info = env.step(action_map[int(action)])

        reward = tensor(reward)
        new_state, new_state_mu, new_state_var = encoder.encode(new_state.to(device))
        new_state = new_state.flatten().unsqueeze(0)

        agent.add_experience(state, action, reward, done, new_state)
        state = new_state
        if done:
            env.reset()


    print('Starting training with {} steps.'.format(cfg['STEPS']))
    mean_step_reward = []
    reward_cur_episode = []
    reward_last_episode = 0
    episode = 1
    start_time = time.time()
    start_time_episode = time.time()

    for steps in range(1, cfg['STEPS'] + 1):

        action = agent.act(state)
        if cfg['RENDER']:
            env.render()
        new_state, reward, done, info = env.step(action_map[int(action)])

        reward = tensor(reward)
        new_state, new_state_mu, new_state_var = encoder.encode(new_state.to(device))
        new_state = new_state.flatten().unsqueeze(0)

        agent.add_experience(state, action, reward, done, new_state)

        #if steps % 4 == 0:
        agent.learn()
        #saver.save(state_mu[3, :], state_var[3, :], action, reward, done)

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
        state_mu = new_state_mu
        state_var = new_state_var

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
