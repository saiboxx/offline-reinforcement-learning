import os
import pickle
import numpy as np
from PIL import Image
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, directory: str):
        self.directory = os.path.join(directory,
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
