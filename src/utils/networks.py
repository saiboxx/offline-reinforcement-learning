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


class DQNCNN(nn.Module):

    def __init__(self, observation_space, action_space):
        super(DQNCNN, self).__init__()
        input_height = observation_space[0]
        input_width = observation_space[1]
        input_channel = observation_space[2]

        kernel_size = 5
        stride = 2

        self.conv1 = nn.Conv2d(input_channel, 16, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride)
        self.bn3 = nn.BatchNorm2d(32)

        def size_out(size, kernel_size=kernel_size, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = size_out(size_out(size_out(input_height)))
        conv_h = size_out(size_out(size_out(input_width)))
        linear_input_size = conv_w * conv_h * 32
        self.head = nn.Linear(linear_input_size, action_space)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.head(x.view(x.size(0), -1))
        return x
