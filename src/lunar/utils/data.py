import os
import pickle
import numpy as np
from typing import Optional
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset


class EnvDataset(Dataset):
    """
    PyTorch Dataset class for trajectories in vector form.
    """

    def __init__(self, root: str):
        """
        Loads dataset to memory and transforms it to tensor.
        :param root: Directory where data files are located
        """
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
        """
        Returns number of samples in dataset.
        :return: number of samples in dataset
        """
        return len(self.rewards) - 1

    def __getitem__(self, idx: int) -> dict:
        """
        Given an index, return a dictionary with the matching tuples.
        :param idx: Index of entry in dataset
        :return: Dict with state, action, reward, done and new state at
        index position
        """
        sample = {
            'state': self.states[idx, :],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'done': self.dones[idx],
            'new_state': self.states[idx + 1, :]
        }
        return sample


class DataSaver(object):
    """
    Saves environment data to disc.
    """
    def __init__(self, directory: str):
        """
        Initializes lists to be saved.
        :param directory: Saving destination
        """
        self.directory = directory
        self.init_dirs()
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def init_dirs(self):
        """
        Create saving directory if it is not existent
        """
        os.makedirs(self.directory, exist_ok=True)

    def save(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: np.ndarray,
             done: np.ndarray):
        """
        Appends passed data to saving list.
        :param state: State
        :param action: Action
        :param reward: Reward
        :param done: Done
        """

        self.states.append(np.expand_dims(state, axis=0))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def close(self):
        """
        Converts lists to numpy format and dumps it as binary file in the specified
        directory
        """
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
    """
    Logs metrics to tensorboard files
    """
    def __init__(self, directory: str, agent_name: str, cfg: Optional[dict]=None):
        """
        Initializes a summary object.
        :param directory: Saving directory of dirs
        :param agent_name: Subfolder for the logs
        :param cfg: Optional dictionary with parameters to be saved.
        """
        self.directory = os.path.join(directory,
                                      agent_name,
                                      datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.directory)
        self.step = 1
        self.episode = 1

        if cfg is not None:
            params = {
                'AGENT': cfg['AGENT'],
                'TRAIN_DATA_PATH': cfg['TRAIN_DATA_PATH'],
                'EPOCHS': int(cfg['EPOCHS']),
                'BATCH_SIZE': int(cfg['BATCH_SIZE']),
                'EVAL_EPISODES': int(cfg['EVAL_EPISODES']),
                'LEARNING_RATE': cfg['LEARNING_RATE'],
                'GAMMA': cfg['GAMMA'],
                'NUM_HEADS': int(cfg['NUM_HEADS']),
                'TARGET_UPDATE_INTERVAL': int(cfg['TARGET_UPDATE_INTERVAL']),
                'SUMMARY_CHECKPOINT': int(cfg['SUMMARY_CHECKPOINT'])
            }
            self.writer.add_hparams(hparam_dict=params, metric_dict={})

    def add_scalar(self, tag: str, value, episode: bool = False):
        """
        Add a scalar to the summary
        :param tag: Tag of scalar
        :param value: Value of scalar
        :param episode: Is the scalar accountable for a step or episode
        """
        step = self.step
        if episode:
            step = self.episode

        self.writer.add_scalar(tag, value, step)

    def adv_step(self):
        """
        Increase step counter
        """
        self.step += 1

    def adv_episode(self):
        """
        Increase episode counter
        """
        self.episode += 1

    def close(self):
        """
        Flush the cached metrics and close writer.
        """
        self.writer.flush()
        self.writer.close()
