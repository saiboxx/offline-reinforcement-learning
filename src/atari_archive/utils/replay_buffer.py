import random
import numpy as np
from typing import Tuple
from torch import tensor
from operator import itemgetter


class ReplayBuffer(object):
    def __init__(self, buffer_size: int, batch_size: int):
        self.max_buffer_size = buffer_size
        self.cur_buffer_size = 0
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []

    def sample(self) -> Tuple:
        if self.cur_buffer_size <= self.batch_size:
            sample_states = self.states
            sample_actions = self.actions
            sample_rewards = self.rewards
            sample_dones = self.dones
            sample_new_states = self.new_states
        else:
            batch_ind = random.sample(range(self.cur_buffer_size), self.batch_size)
            sample_states = itemgetter(*batch_ind)(self.states)
            sample_actions = itemgetter(*batch_ind)(self.actions)
            sample_rewards = itemgetter(*batch_ind)(self.rewards)
            sample_dones = itemgetter(*batch_ind)(self.dones)
            sample_new_states = itemgetter(*batch_ind)(self.new_states)

        return sample_states, sample_actions, sample_rewards, sample_dones, sample_new_states

    def add(self, state: tensor,
            action: tensor,
            reward: tensor,
            done: bool,
            new_state: tensor):

        if self.cur_buffer_size >= self.max_buffer_size:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.new_states[0]

        self.states.append(state)
        self.actions.append(action.flatten())
        self.rewards.append(reward)
        self.dones.append(done)
        self.new_states.append(new_state)
        self.cur_buffer_size = len(self.states)

class ReplayBuffer(object):
    def __init__(self, buffer_size: int, batch_size: int):
        self.max_buffer_size = buffer_size
        self.cur_buffer_size = 0
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []

    def sample(self) -> Tuple:
        if self.cur_buffer_size <= self.batch_size:
            sample_states = self.states
            sample_actions = self.actions
            sample_rewards = self.rewards
            sample_dones = self.dones
            sample_new_states = self.new_states
        else:
            batch_ind = random.sample(range(self.cur_buffer_size), self.batch_size)
            sample_states = itemgetter(*batch_ind)(self.states)
            sample_actions = itemgetter(*batch_ind)(self.actions)
            sample_rewards = itemgetter(*batch_ind)(self.rewards)
            sample_dones = itemgetter(*batch_ind)(self.dones)
            sample_new_states = itemgetter(*batch_ind)(self.new_states)

        return sample_states, sample_actions, sample_rewards, sample_dones, sample_new_states

    def add(self, state: tensor,
            action: tensor,
            reward: tensor,
            done: bool,
            new_state: tensor):

        if self.cur_buffer_size >= self.max_buffer_size:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.dones[0]
            del self.new_states[0]

        self.states.append(state)
        self.actions.append(action.flatten())
        self.rewards.append(reward)
        self.dones.append(done)
        self.new_states.append(new_state)
        self.cur_buffer_size = len(self.states)

