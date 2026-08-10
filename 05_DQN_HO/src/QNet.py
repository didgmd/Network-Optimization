import torch.nn as nn


# 定义Q网络
class QNet(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        actions_tensor = self.fc2(x)
        return actions_tensor


# 定义Multiple Heads Q网络
class MultipleHeadsQNet(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions_a2, n_actions_a4):
        super(MultipleHeadsQNet, self).__init__()

        # 定义共享层
        self.shared_layer = nn.Sequential(nn.Linear(n_states, n_hidden), nn.ReLU())

        # 定义HeadA2
        self.headA2 = nn.Linear(n_hidden, n_actions_a2)

        # 定义HeadA4
        self.headA4 = nn.Linear(n_hidden, n_actions_a4)

    def forward(self, x):
        x = self.shared_layer(x)
        actions_tensor_a2 = self.headA2(x)
        actions_tensor_a4 = self.headA4(x)
        return actions_tensor_a2, actions_tensor_a4


class QNet3(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(QNet3, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        actions_tensor = self.fc3(x)
        return actions_tensor
