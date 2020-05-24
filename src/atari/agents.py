import os
import numpy as np
import random
from typing import Union
import torch
from torch import tensor
from torch.optim import Adam
from torch.nn import SmoothL1Loss
import torchsummary
from src.atari.utils.networks import DQNCNN, DQNCNNLight, DQNDense, DQNLSTM
from src.atari.utils.replay_buffer import ReplayBuffer
from src.atari.utils.data import Summary
from torchvision.transforms import Compose, ToPILImage, Grayscale, Resize, ToTensor


class Agent(object):
    def __init__(self, observation_space: int, action_space: int):
        self.observation_space = observation_space
        self.action_space = action_space
        self.name = None
        self.summary_writer = None

    def add_summary_writer(self, summary_writer: Summary):
        self.summary_writer = summary_writer

    def act(self, state: tensor) -> tensor:
        pass

    def add_experience(self,
                       state: tensor,
                       action: tensor,
                       reward: tensor,
                       done: bool,
                       new_state: tensor):
        pass

    def learn(self, *args):
        pass

    def print_model(self):
        pass

    def save(self, *args):
        pass

    def __repr__(self):
        return self.name


class RandomAgent(Agent):
    def __init__(self, observation_space: int, action_space: int):
        super().__init__(observation_space, action_space)
        self.name = 'RandomAgent'

    def act(self, state: np.ndarray) -> np.ndarray:
        return random.randint(0, self.action_space - 1)


class DQNAgent(Agent):
    def __init__(self, observation_space: int, action_space: int):
        super().__init__(observation_space, action_space)
        self.name = 'DQNAgent'
        self.summary_checkpoint = 1000

        self.target_update_steps = 500
        self.input_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Utilizing device {}'.format(self.device))
        self.policy = DQNLSTM(self.input_size, action_space).to(self.device)
        self.target = DQNLSTM(self.input_size, action_space).to(self.device)
        self.target.load_state_dict(self.target.state_dict())
        self.target.eval()
        self.optimizer = Adam(self.policy.parameters(), lr=0.00025)
        self.loss = SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(buffer_size=50000, batch_size=32)

        self.steps_done = 0
        self.eps_start = 0.99
        self.eps_end = 0.05
        self.eps_decay = (self.eps_start - self.eps_end) / 200000
        self.eps_threshold = self.eps_start

    def act(self, state: tensor) -> tensor:
        if self.eps_threshold > self.eps_end:
            self.eps_threshold -= self.eps_decay
        if torch.rand(1) > self.eps_threshold:
            with torch.no_grad():
                self.policy.eval()
                return torch.argmax(self.policy(state.to(self.device)))
        else:
            return tensor(random.randint(0, self.action_space - 1))

    def add_experience(self,
                       state: tensor,
                       action: tensor,
                       reward: tensor,
                       done: bool,
                       new_state: tensor):
        self.replay_buffer.add(state, action, reward, done, new_state)

    def learn(self):
        self.policy.train()
        state, action, reward, done, new_state = self.replay_buffer.sample()

        state = torch.cat(state).to(self.device)
        action = torch.stack(action).to(self.device)
        reward = torch.stack(reward).to(self.device)
        done = torch.tensor(done).bool().to(self.device)
        new_state = torch.cat(new_state).to(self.device)

        # Get Q(s,a) for actions taken
        state_action_values = self.policy(state).gather(1, action)

        # Get V(s') for the new states w/ mask for final state
        next_state_values, _ = torch.max(self.target(new_state).detach(), dim=1)
        next_state_values[done] = 0

        # Get expected Q values
        expected_state_action_values = (next_state_values * 0.99) + reward

        # Compute loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(-1))

        # Optimize w/ Clipping
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target every n steps
        if self.steps_done % self.target_update_steps == 0:
            self.target.load_state_dict(self.policy.state_dict())

        # Log metrics
        if self.summary_writer is not None \
                and self.steps_done % self.summary_checkpoint == 0:
            self.summary_writer.add_scalar('Loss', loss)
            self.summary_writer.add_scalar('Expected Q Values', expected_state_action_values.mean())

        self.steps_done += 1

    def print_model(self):
        #torchsummary.summary(self.policy, input_size=(self.input_size,))
        pass

    def save(self):
        directory = os.path.join('models',
                                 self.name,
                                 'model.pt')
        os.makedirs('models/' + self.name, exist_ok=True)
        torch.save(self.target, directory)


class DoubleDQNAgent(DQNAgent):
    def __init__(self, observation_space: int, action_space: int):
        super().__init__(observation_space, action_space)
        self.name = 'DoubleDQNAgent'
        self.tau = 0.01

    def learn(self):
        self.policy.train()
        state, action, reward, done, new_state = self.replay_buffer.sample()

        state = torch.cat(state).to(self.device)
        action = torch.stack(action).to(self.device)
        reward = torch.stack(reward).to(self.device)
        done = torch.tensor(done).bool().to(self.device)
        new_state = torch.cat(new_state).to(self.device)

        # Get Q(s,a) for actions taken
        state_action_values = self.policy(state).gather(1, action)

        with torch.no_grad():
            # Get action for next state from greedy policy
            new_action = torch.argmax(self.policy(new_state), dim=1)

            # Get V(s') for the new states w/ action decided by policy w/ mask for final state
            next_state_values = self.target(new_state).gather(1, new_action.unsqueeze(1))
            next_state_values[done] = 0

        # Get expected Q values
        expected_state_action_values = (next_state_values.squeeze() * 0.99) + reward

        # Compute loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(-1))

        # Optimize w/ Clipping
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target every n steps
        if self.steps_done % self.target_update_steps == 0:
            self.target.load_state_dict(self.policy.state_dict())

        # Log metrics
        if self.summary_writer is not None \
                and self.steps_done % self.summary_checkpoint == 0:
            self.summary_writer.add_scalar('Loss', loss)
            self.summary_writer.add_scalar('Expected Q Values', expected_state_action_values.mean())

        self.steps_done += 1


class OfflineDQNAgent(Agent):
    def __init__(self, observation_space: int, action_space: int):
        super().__init__(observation_space, action_space)
        self.name = 'OfflineDQNAgent'
        self.summary_checkpoint = 1000
        self.batches_done = 0

        self.target_update_steps = 5000
        self.input_size = 16

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Utilizing device {}'.format(self.device))
        self.policy = DQNDense(self.input_size, action_space).to(self.device)
        self.target = DQNDense(self.input_size, action_space).to(self.device)
        self.target.load_state_dict(self.target.state_dict())
        self.target.eval()
        self.optimizer = Adam(self.policy.parameters(), lr=0.00025)
        self.loss = SmoothL1Loss()

    def act(self, state: tensor) -> tensor:
        with torch.no_grad():
            self.policy.eval()
            return torch.argmax(self.policy(state.to(self.device)))

    def learn(self, batch: dict):

        state = batch['state'].to(self.device, non_blocking=True)
        action = batch['action'].to(self.device, non_blocking=True)
        reward = batch['reward'].to(self.device, non_blocking=True)
        done = batch['done'].to(self.device, non_blocking=True)
        new_state = batch['new_state'].to(self.device, non_blocking=True)

        # Get Q(s,a) for actions taken
        state_action_values = self.policy(state).gather(1, action.unsqueeze(-1))

        # Get V(s') for the new states w/ mask for final state
        next_state_values, _ = torch.max(self.target(new_state).detach(), dim=1)
        next_state_values[done] = 0

        # Get expected Q values
        expected_state_action_values = (next_state_values * 0.999) + reward

        # Compute loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(-1))

        # Optimize w/ Clipping
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target every n steps
        if self.batches_done % self.target_update_steps == 0:
            self.target.load_state_dict(self.policy.state_dict())

        # Log metrics
        if self.summary_writer is not None \
                and self.batches_done % self.summary_checkpoint == 0:
            self.summary_writer.add_scalar('Loss', loss)
            self.summary_writer.add_scalar('Expected Q Values', expected_state_action_values.mean())

        self.batches_done += 1

    def print_model(self):
        torchsummary.summary(self.policy, input_size=(self.input_size,))

    def save(self, epoch: int):
        directory = os.path.join('models',
                                 self.name,
                                 str(epoch) + '.pt')
        os.makedirs('models/' + self.name, exist_ok=True)
        torch.save(self.target, directory)
