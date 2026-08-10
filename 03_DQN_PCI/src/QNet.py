import torch.nn as nn

# import torch.nn.functional as F


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
