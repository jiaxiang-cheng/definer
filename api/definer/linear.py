import torch
import torch.nn as nn
import torch.optim as optim


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One output for regression

    def forward(self, x):
        return self.linear(x)
