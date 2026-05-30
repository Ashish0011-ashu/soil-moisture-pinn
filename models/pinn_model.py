# neural network class
import torch
import torch.nn as nn

class PINN(nn.Module):

    def __init__(self):
        super(PINN, self).__init__()

        self.net = nn.Sequential(

            nn.Linear(5, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)