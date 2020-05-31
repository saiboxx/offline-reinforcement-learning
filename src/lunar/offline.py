import gym
import yaml
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.lunar.utils.data import EnvDataset, Summary
from src.lunar.agents import Agent, OfflineDQNAgent, EnsembleOffDQNAgent, REMOffDQN


def main():
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    train(cfg)


def train(cfg: dict):
    print('Loading environment LunarLander-v2.')
    env = gym.make('LunarLander-v2')
    env.reset()
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    state = np.zeros(observation_space)

    print('Creating Agent.')
    agent = create_agent(observation_space, action_space, cfg)
    summary = Summary(cfg['SUMMARY_PATH'], agent.name, cfg)
    agent.print_model()
    agent.add_summary_writer(summary)

    print('Initializing Dataloader.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Utilizing device {}'.format(device))
    training_data = EnvDataset(cfg['TRAIN_DATA_PATH'])
    data_loader = DataLoader(dataset=training_data,
                             batch_size=cfg['BATCH_SIZE'],
                             shuffle=True,
                             num_workers=8,
                             pin_memory=True)

    print('Start training with {} epochs'.format(cfg['EPOCHS']))
    for e in range(1, cfg['EPOCHS'] + 1):
        for i_batch, sample_batched in enumerate(tqdm(data_loader)):
            agent.learn(sample_batched)

            summary.adv_step()

        rewards = []
        mean_reward = []
        counter = 0
        while counter < cfg['EVAL_EPISODES']:
            action = agent.act(state)
            if cfg['EVAL_RENDER']:
                env.render()
            state, reward, done, _ = env.step(int(action))
            rewards.append(reward)
            if done:
                env.reset()
                mean_reward.append(sum(rewards))
                rewards = []
                counter += 1

        agent.save(e)
        summary.add_scalar('Episode Mean Reward', np.mean(mean_reward), True)
        summary.adv_episode()
        summary.writer.flush()

    print('Closing environment.')
    env.close()


def create_agent(observation_space: int, action_space: int, cfg: dict) -> Agent:
    if cfg['AGENT'] == 'DQN':
        return OfflineDQNAgent(observation_space, action_space, cfg)
    elif cfg['AGENT'] == 'ENSEMBLE':
        return EnsembleOffDQNAgent(observation_space, action_space, cfg)
    elif cfg['AGENT'] == 'REM':
        return REMOffDQN(observation_space, action_space, cfg)
    else:
        print('No valid agent with name {} found. Exiting...'.format(cfg['AGENT']))
        exit()


if __name__ == '__main__':
    main()
