import torch
from torchvision.models import resnet18

from nenepy_summary import TorchSummary

resnet = resnet18(pretrained=False)

size_mode = False
if size_mode:
    summary = TorchSummary(model=resnet, batch_size=5, is_train=True)
    output = summary.forward_size([3, 256, 256])

    print(output.shape)

else:
    summary = TorchSummary(model=resnet, is_train=True)
    input_tensor = torch.rand(size=[2, 3, 256, 256])
    output = summary.forward_tensor(input_tensor)

    print(output.shape)
