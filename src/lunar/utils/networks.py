from torch import nn, tensor


class DQNDense(nn.Module):
    def __init__(self, observation_space: int, action_space: int):
        super(DQNDense, self).__init__()
        self.fc1 = nn.Linear(in_features=observation_space, out_features=128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(in_features=32, out_features=action_space)

        self.elu = nn.ELU()

    def forward(self, x: tensor) -> tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.fc3(x)
        return x


class DQNMultiHead(nn.Module):
    def __init__(self, observation_space: int, action_space: int, num_heads: int):
        super(DQNMultiHead, self).__init__()
        self.num_heads = num_heads
        self.action_space = action_space

        self.fc1 = nn.Linear(in_features=observation_space, out_features=128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.bn2 = nn.BatchNorm1d(32)

        self.heads = nn.Conv1d(in_channels=32 * num_heads,
                               out_channels=action_space * num_heads,
                               kernel_size=1,
                               groups=num_heads)

        self.elu = nn.ELU()

    def forward(self, x: tensor) -> tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu(x)

        x = x.repeat(1, self.num_heads).unsqueeze(-1)
        x = self.heads(x)
        return x.view(-1, self.action_space, self.num_heads).permute(2, 0, 1)


class QRDQNDense(nn.Module):
    def __init__(self, observation_space: int, action_space: int, num_quant: int):
        super(QRDQNDense, self).__init__()
        self.action_space = action_space
        self.num_quant = num_quant
        self.fc1 = nn.Linear(in_features=observation_space, out_features=128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(in_features=32, out_features=action_space * self.num_quant)

        self.elu = nn.ELU()

    def forward(self, x: tensor) -> tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.fc3(x)
        return x.view(-1, self.action_space, self.num_quant)