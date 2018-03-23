import torch.nn as nn
import torch
from torch.autograd import Variable
from ModelParallel import ModelParallel
import torchvision.models as models
import time


class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()

        self.linear = nn.Sequential(nn.Linear(30, 1000),
                                    nn.Sigmoid(),
                                    nn.Linear(1000, 3000),
                                    nn.Sigmoid(),
                                    nn.Linear(3000, 2)
                                    )

    def forward(self, input):
        return self.linear(input)


class Ensemble(nn.Module):
    def __init__(self, m, mode='small'):
        super(Ensemble, self).__init__()
        if mode == 'small':
            self.module = nn.ModuleList([SmallNet() for i in range(m)])
        elif mode == 'large':
            self.module = nn.ModuleList([models.resnet50() for i in range(m)])

    def forward(self, input):
        return [module(input) for module in self.module]


def test_model_parallel(mode='small'):
    ensemble = Ensemble(4, mode)

    model = ModelParallel(ensemble, device_ids=[0, 1, 2, 3], output_device=0)
    if mode == 'small':
        input = Variable(torch.rand(512, 30))
    elif mode == 'large':
        input = Variable(torch.rand(128, 3, 224, 224))

    end = time.time()
    y = model(input)
    print('using model parallel')
    print('time : ', time.time() - end)


def test_without_parallel(mode='small'):
    ensemble = Ensemble(4, mode)
    [ensemble.module[i].cuda(i) for i in range(4)]

    if mode == 'small':
        input = Variable(torch.rand(512, 30))
    elif mode == 'large':
        input = Variable(torch.rand(128, 3, 224, 224))

    end = time.time()
    y = [ensemble.module[i](input.cuda(i)) for i in range(4)]
    print('without model parallel')
    print('time: ', time.time() - end)


if __name__ == '__main__':
    # on small net
    test_model_parallel('small')
    test_without_parallel('small')

    # on imagenet resnet50
    test_model_parallel('large')
    test_without_parallel('large')
