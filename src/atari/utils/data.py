import os
import pickle
import psutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import torch
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class EnvDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.states_mu = torch.load(self.root + '/state_mu.pt')
        self.states_var = torch.load(self.root + '/state_var.pt')
        self.actions = torch.load(self.root + '/action.pt')
        self.rewards = torch.load(self.root + '/reward.pt')
        self.dones = torch.load(self.root + '/done.pt')

    def __len__(self) -> int:
        return len(self.actions) - 1

    def __getitem__(self, idx: int) -> dict:
        state = self.get_normal_sample(idx)
        new_state = self.get_normal_sample(idx + 1)

        sample = {
            'state': state,
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'done': self.dones[idx],
            'new_state': new_state
        }
        return sample

    def get_normal_sample(self, idx: int):
        std = torch.exp(0.5 * self.states_var[idx, :])
        eps = torch.randn_like(std)
        return self.states_mu[idx, :] + eps * std


class VAEDataset(Dataset):

    def __init__(self, root: str):
        self.root = root
        self.state = torch.load(self.root + '/autoencoder_data.pt')

    def __len__(self) -> int:
        return len(self.state)

    def __getitem__(self, idx: int) -> dict:
        return self.state[idx]


class DataSaver(object):
    def __init__(self, directory: str):
        self.directory = directory
        self.init_dirs()
        self.states_mu = []
        self.states_var = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def init_dirs(self):
        os.makedirs(self.directory, exist_ok=True)

    def save(self,
             state_mu: tensor,
             state_var: tensor,
             action: tensor,
             reward: tensor,
             done: np.ndarray):

        self.states_mu.append(state_mu)
        self.states_var.append(state_var)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def close(self):
        states_mu = torch.stack(self.states_mu)
        states_var = torch.stack(self.states_var)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)
        dones = torch.tensor(self.dones).bool()

        torch.save(states_mu, self.directory + '/state_mu.pt')
        torch.save(states_var, self.directory + '/state_var.pt')
        torch.save(actions, self.directory + '/action.pt')
        torch.save(rewards, self.directory + '/reward.pt')
        torch.save(dones, self.directory + '/done.pt')


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


###############################################################################
# ARCHIVE                                                                     #
###############################################################################
class ImgDataSaver(object):
    def __init__(self, directory: str):
        self.directory = directory
        self.init_dirs()
        self.max_buffer = 50000
        self.buffer = []
        self.counter = 0
        self.actions = []
        self.rewards = []
        self.dones = []

    def init_dirs(self):
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.directory + '/state', exist_ok=True)
        os.makedirs(self.directory + '/action', exist_ok=True)
        os.makedirs(self.directory + '/reward', exist_ok=True)
        os.makedirs(self.directory + '/done', exist_ok=True)

    def save(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: np.ndarray,
             done: np.ndarray):
        state = Image.fromarray(np.uint8(state)).resize((110, 84)).convert('L')
        self.buffer.append(state)

        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        if len(self.buffer) >= self.max_buffer:
            self.flush()

    def flush(self):
        for state in self.buffer:
            state.save(self.directory + '/state/' + str(self.counter) + '.jpg')
            self.counter += 1
        self.buffer = []

    def close(self):
        self.flush()
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        with open(self.directory + '/action/' + 'action.pkl', 'wb') as output:
            pickle.dump(actions, output)
        with open(self.directory + '/reward/' + 'reward.pkl', 'wb') as output:
            pickle.dump(rewards, output)
        with open(self.directory + '/done/' + 'done.pkl', 'wb') as output:
            pickle.dump(dones, output)


class ImgEnvDataset(Dataset):

    def __init__(self, root: str):
        self.root = root
        with open(self.root + '/action/' + 'action.pkl', 'rb') as file:
            self.actions = pickle.load(file)
        with open(self.root + '/reward/' + 'reward.pkl', 'rb') as file:
            self.rewards = pickle.load(file)
        with open(self.root + '/done/' + 'done.pkl', 'rb') as file:
            self.dones = pickle.load(file)
        self.transform = Compose([ToTensor()])

        # Loads states to memory until RAM is x percent full
        # to counter the I/O bottleneck
        self.max_ram_usage = 90
        self.states = []
        self.idx = 0
        self.load_states_to_RAM()

    def load_states_to_RAM(self):
        idx = 0
        print('Loading data until RAM is {} % full.'.format(self.max_ram_usage))
        while psutil.virtual_memory()[2] < self.max_ram_usage:
            state_img = os.path.join(self.root,
                                     'state',
                                     str(idx) + '.jpg')

            state = self.transform(Image.open(state_img))
            self.states.append(state)

            self.idx += 1
            if self.idx % 25000 == 0:
                print('RAM is {} full.'.format(psutil.virtual_memory()[2]))

        print('Loaded {} states to memory.'.format(self.idx))

    def __len__(self) -> int:
        return len(self.rewards) - 1

    def __getitem__(self, idx: int) -> dict:

        if (idx + 1) < self.idx:
            state = self.states[idx]
            new_state = self.states[idx + 1]
        else:
            state_img = os.path.join(self.root,
                                     'state',
                                     str(idx) + '.jpg')
            new_state_img = os.path.join(self.root,
                                         'state',
                                         str(idx + 1) + '.jpg')

            state = self.transform(Image.open(state_img))
            new_state = self.transform(Image.open(new_state_img))

        sample = {
            'state': state,
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'done': self.dones[idx],
            'new_state': new_state
        }
        return sample


class ImgEnvDatasetInMemory(Dataset):

    def __init__(self, root: str):
        self.root = root
        with open(self.root + '/action/' + 'action.pkl', 'rb') as file:
            self.actions = pickle.load(file)
        with open(self.root + '/reward/' + 'reward.pkl', 'rb') as file:
            self.rewards = pickle.load(file)
        with open(self.root + '/done/' + 'done.pkl', 'rb') as file:
            self.dones = pickle.load(file)
        self.transform = Compose([ToTensor()])

        self.states = []
        for idx in tqdm(range(len(self.rewards))):
            state_img = os.path.join(self.root,
                                     'state',
                                     str(idx) + '.jpg')
            state_img = self.transform(Image.open(state_img))
            self.states.append(state_img)

    def __len__(self) -> int:
        return len(self.rewards) - 1

    def __getitem__(self, idx: int) -> dict:
        sample = {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'done': self.dones[idx],
            'new_state': self.states[idx + 1]
        }
        return sample