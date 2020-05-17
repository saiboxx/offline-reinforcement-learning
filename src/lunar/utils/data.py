import os
import pickle
import numpy as np
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset


class EnvDataset(Dataset):

    def __init__(self, root: str):
        self.root = root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(self.root + '/state.pkl', 'rb') as file:
            self.states = pickle.load(file)
        with open(self.root + '/action.pkl', 'rb') as file:
            self.actions = pickle.load(file)
        with open(self.root + '/reward.pkl', 'rb') as file:
            self.rewards = pickle.load(file)
        with open(self.root + '/done.pkl', 'rb') as file:
            self.dones = pickle.load(file)

        self.states = torch.tensor(self.states).float()
        self.actions = torch.tensor(self.actions)
        self.rewards = torch.tensor(self.rewards).float()
        self.dones = torch.tensor(self.dones).bool()

    def __len__(self) -> int:
        return len(self.rewards) - 1

    def __getitem__(self, idx: int) -> dict:
        sample = {
            'state': self.states[idx, :],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'done': self.dones[idx],
            'new_state': self.states[idx + 1, :]
        }
        return sample


class DataSaver(object):
    def __init__(self, directory: str):
        self.directory = directory
        self.init_dirs()
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def init_dirs(self):
        os.makedirs(self.directory, exist_ok=True)

    def save(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: np.ndarray,
             done: np.ndarray):

        self.states.append(np.expand_dims(state, axis=0))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def close(self):
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        with open(self.directory + '/state.pkl', 'wb') as output:
            pickle.dump(states, output)
        with open(self.directory + '/action.pkl', 'wb') as output:
            pickle.dump(actions, output)
        with open(self.directory + '/reward.pkl', 'wb') as output:
            pickle.dump(rewards, output)
        with open(self.directory + '/done.pkl', 'wb') as output:
            pickle.dump(dones, output)


class Summary(object):
    def __init__(self, directory: str, agent_name: str):
        self.directory = os.path.join(directory,
                                      agent_name,
                                      datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.directory)
        self.step = 1
        self.episode = 1

    def add_scalar(self, tag: str, value, episode: bool = False):
        step = self.step
        if episode:
            step = self.episode

        self.writer.add_scalar(tag, value, step)

    def adv_step(self):
        self.step += 1

    def adv_episode(self):
        self.episode += 1

    def close(self):
        self.writer.flush()
        self.writer.close()
