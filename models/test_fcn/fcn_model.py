import torch
from torch import nn
from torch.functional import F


class LeNet300(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, output_dim)
        # self.dropout = nn.Dropout(dropout_rate)
        # activation function for hidden layers
        self.act_fn = F.relu

    def forward(self, x):
        self.activations = {}

        out = torch.flatten(x, 1)
        out = self.act_fn(self.fc1(out))
        self.activations["relu1"] = out
        # out = self.dropout(out)

        out = self.act_fn(self.fc2(out))
        self.activations["relu2"] = out
        # out = self.dropout(out)

        out = self.act_fn(self.fc3(out))
        self.activations["relu3"] = out
        # out = self.dropout(out)

        out = self.fc4(out)
        return out

    def load_pretrained_weights(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
        print(f"Model loaded from {checkpoint_path}")
