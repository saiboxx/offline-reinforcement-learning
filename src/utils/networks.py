from torch import nn


class DQN(nn.Module):

    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=observation_space, out_features=128)
        self.bn1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.bn2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(in_features=32, out_features=action_space)

        self.elu = nn.ELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.fc3(x)
        return x
