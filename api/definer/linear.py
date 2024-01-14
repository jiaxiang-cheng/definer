import torch
import torch.nn as nn
import torch.optim as optim


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One output for regression

    def forward(self, x):
        return self.linear(x)


# class LinearPositiveRegressionModel(nn.Module):
#     def __init__(self, input_size):
#         super(LinearPositiveRegressionModel, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(1, input_size))
#         # self.relu = nn.ReLU()
#
#     def forward(self, x):
#         return nn.functional.linear(x, self.weight)


class LinearPositiveRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearPositiveRegressionModel, self).__init__()
        self.log_weight = nn.Parameter(torch.Tensor(1, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, x):
        return nn.functional.linear(x, self.log_weight.exp())
