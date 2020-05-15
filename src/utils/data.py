import os
import pickle
import numpy as np
from PIL import Image
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class EnvironmentDataset(Dataset):

    def __init__(self, root: str):
        self.root = root
        with open(self.root + '/action/' + 'action.pkl', 'rb') as file:
            self.actions = pickle.load(file)
        with open(self.root + '/reward/' + 'reward.pkl', 'rb') as file:
            self.rewards = pickle.load(file)
        with open(self.root + '/done/' + 'done.pkl', 'rb') as file:
            self.dones = pickle.load(file)
        self.transform = Compose([ToTensor()])

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(self, idx: int) -> dict:
        state_img = os.path.join(self.root,
                                 'state',
                                 str(idx) + '.jpg')
        new_state_img = os.path.join(self.root,
                                     'new_state',
                                     str(idx) + '.jpg')

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


class DataSaver(object):
    def __init__(self, directory: str):
        self.directory = directory
        self.init_dirs()
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
        os.makedirs(self.directory + '/new_state', exist_ok=True)

    def save(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: np.ndarray,
             done: np.ndarray,
             new_state: np.ndarray):
        state = Image.fromarray(np.uint8(state)).resize((110, 84)).convert('L')
        new_state = Image.fromarray(np.uint8(new_state)).resize((110, 84)).convert('L')

        state.save(self.directory + '/state/' + str(self.counter) + '.jpg')
        new_state.save(self.directory + '/new_state/' + str(self.counter) + '.jpg')

        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        self.counter += 1

    def close(self):
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        with open(self.directory + '/action/' + 'action.pkl', 'wb') as output:
            pickle.dump(actions, output)
        with open(self.directory + '/reward/' + 'reward.pkl', 'wb') as output:
            pickle.dump(rewards, output)
        with open(self.directory + '/done/' + 'done.pkl', 'wb') as output:
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
