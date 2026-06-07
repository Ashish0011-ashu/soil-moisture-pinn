import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        # Keeping Tanh but adding LayerNorm to prevent gradient flatlining
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),   # Keeps values in the sharp, learning region of Tanh
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Tanh(),

            nn.Linear(64, 1)    # Raw output for regression
        )

    def forward(self, x):
        return self.net(x)