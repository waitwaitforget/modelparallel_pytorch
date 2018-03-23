import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather


def replicate(module, device):
    return module.cuda(device)


class ModelParallel(nn.Module):

    def __init__(self, model, device_ids=None, output_device=None):
        super(ModelParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = model
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
            if not hasattr(model, 'module'):
                raise ValueError("model does not has module attribute")
            if len(device_ids) < len(model.module):
                print('warning: number of devices is not enough for module parallel')
            else:
                device_ids = device_ids[:len(model.module)]

        if output_device is None:
            output_device = device_ids[0]
        self.device_ids = device_ids
        self.module = model.module  # module is a list

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(input, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return [replicate(m, id) for m, id in zip(module, device_ids)]

    def scatter(self, inputs, kwargs, device_ids):
        inputs = [input.cuda(id) for input, id in zip(inputs, device_ids)]
        kwargs = kwargs * device_ids
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device)


def unittest():
    from torch.autograd import Variable
    x = torch.rand(4, 3)
    x = Variable(x)

    def test(*inputs):
        inputs, kwargs = scatter_kwargs(inputs, None, [0])
        print(inputs)
    test(x)


if __name__ == '__main__':
    unittest()
