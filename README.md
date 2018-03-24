# Model Parallelism
Model Parallelism for pytorch training multiple networks on multiple GPUs.

## ToDo List
- [ ] Handle different kwargs for different networks

# Usage
Model parallel is a wrapper for training multiple networks on multi-GPU simultaneously. Such as training ensemble models or multiple choice learning networks.

Unlike data parallel, the outputs of model parallel is a list for general purpose.

```python
# First define a ensemble module
import torch
import torch.nn as nn
import torchvision.models as models
from ModelParallel import ModelParallel


class Ensemble(nn.Module):
    def __init__(self, m):
        super(Ensemble, self).__init__()
        self.m = m
        self.module = nn.ModuleList([models.resnet50() for _ in range(m)])

    def forward(self, input):
        return [self.module[i](input) for i in range(self.m)]

model = Ensemble(4)
model = ModelParallel(model, device_ids=[0, 1, 2, 3], output_device=0)

x = torch.rand(128, 3, 224, 224)
y = model(Variable(x))

```

## Useful links
Some multithreading code is borrowed from [pytorch data parallel](https://github.com/pytorch/pytorch/blob/v0.3.1/torch/nn/parallel/parallel_apply.py)
