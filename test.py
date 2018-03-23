import torch.nn as nn
import torch
from torch.autograd import Variable
from ModelParallel import ModelParallel


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear = nn.Linear(3, 10)

    def forward(self, input):
        return self.linear(input)


class Ensemble(nn.Module):
    def __init__(self, m):
        super(Ensemble, self).__init__()

        self.module = nn.ModuleList([Net() for i in range(m)])

    def forward(self, input):
        return [module(input) for module in self.module]


model = Ensemble(3)

model = ModelParallel(model, device_ids=[0, 1, 2], output_device=0)
print model
input = Variable(torch.rand(10, 3))
print model(input)
