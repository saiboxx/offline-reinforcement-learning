import gym
import yaml
import multiprocessing
import numpy as np
import torch
from src.lunar.offline import create_agent


def main():
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    for i in range(cfg['NUM_ENVS']):
        p = multiprocessing.Process(target=run, args=(i, cfg))
        p.start()


def run(worker_id: int, cfg: dict):
    env = gym.make('LunarLander-v2')
    env.reset()
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    state = np.zeros(observation_space)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = create_agent(observation_space, action_space, cfg)
    agent.policy = torch.load(cfg['INF_MODEL'])
    agent.policy.to(device)

    rewards = []
    while True:
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(int(action))
        rewards.append(reward)

        if done:
            env.reset()
            print('Worker {0}; Episode Reward: {1:4.2f}'.format(worker_id, sum(rewards)))
            rewards = []


if __name__ == '__main__':
    main()
