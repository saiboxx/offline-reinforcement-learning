import numpy as np
import random
import torch
from torch.optim import Adam
from torch.nn import SmoothL1Loss
from src.utils.networks import DQN
from src.utils.replay_buffer import ReplayBuffer


class Agent(object):
    def __init__(self, observation_space: int, action_space: int):
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, state: np.ndarray) -> np.ndarray:
        pass

    def add_experience(self,
                       state: np.ndarray,
                       action: np.ndarray,
                       reward: np.ndarray,
                       done: np.ndarray,
                       new_state: np.ndarray):
        pass

    def learn(self):
        pass


class RandomAgent(Agent):
    def __init__(self, observation_space: int, action_space: int):
        super().__init__(observation_space, action_space)

    def act(self, state: np.ndarray) -> np.ndarray:
        return random.randint(0, self.action_space - 1)


class DQNAgent(Agent):
    def __init__(self, observation_space: int, action_space: int):
        super().__init__(observation_space, action_space)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Utilizing device {}'.format(self.device))
        self.policy = DQN(observation_space, action_space).to(self.device)
        self.target = DQN(observation_space, action_space).to(self.device)
        self.target.load_state_dict(self.target.state_dict())
        self.target.eval()
        self.optimizer = Adam(self.policy.parameters(), lr=0.001)
        self.loss = SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_size=100000, batch_size=128)
        self.steps_done = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000

    def act(self, state: np.ndarray) -> np.ndarray:
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if torch.rand(1) > eps_threshold:
            with torch.no_grad():
                state = torch.from_numpy(state).to(self.device).float()
                action = torch.argmax(self.policy(state))
                return action.cpu().detach().numpy()
        else:
            return np.asarray(random.randint(0, self.action_space - 1))

    def add_experience(self,
                       state: np.ndarray,
                       action: np.ndarray,
                       reward: np.ndarray,
                       done: bool,
                       new_state: np.ndarray):
        self.replay_buffer.add(state, action, reward, done, new_state)

    def learn(self):
        state, action, reward, done, new_state = self.replay_buffer.sample()

        state = torch.tensor(state).float().to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).bool().to(self.device)
        new_state = torch.tensor(new_state).float().to(self.device)

        # Get Q(s,a) for actions taken
        state_action_values = self.policy(state).gather(1, action)

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
        if self.steps_done % 2000 == 0:
            self.target.load_state_dict(self.policy.state_dict())
