import torch.nn as nn
import torch
from torch.autograd import Variable
from ModelParallel import ModelParallel


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear = nn.Sequential(nn.Linear(3, 100),
                                    nn.Sigmoid(),
                                    nn.Linear(100, 300),
                                    nn.Sigmoid(),
                                    nn.Linear(300, 2)
                                    )

    def forward(self, input):
        return self.linear(input)


class Ensemble(nn.Module):
    def __init__(self, m):
        super(Ensemble, self).__init__()

        self.module = nn.ModuleList([Net() for i in range(m)])

    def forward(self, input):
        return [module(input) for module in self.module]


ensemble = Ensemble(5)

model = ModelParallel(ensemble, device_ids=[0, 1, 2, 4, 5], output_device=0)
print len(model.module.module)
input = Variable(torch.rand(10, 3))

import time
end = time.time()
y = model(input)
print('using model parallel')
print('time : ', time.time() - end)

end = time.time()
y = ensemble(input)
print('without model parallel')
print('time: ', time.time() - end)
