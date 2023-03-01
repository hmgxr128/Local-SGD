import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, RandomSampler


pth = '/home/guxinran/localsgd/post_local/data'
tsf = transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

dataset = torchvision.datasets.CIFAR10(root=pth, train=True, download=False, transform=tsf)

sampler = RandomSampler(dataset, replacement=True, num_samples=20)


dataloader = DataLoader(dataset, batch_size=10, pin_memory=True, sampler=sampler, num_workers=2, drop_last=1)

for idx, data in enumerate(dataloader):
    print(list(sampler))
sampler = RandomSampler(dataset, replacement=True, num_samples=20)

for idx, data in enumerate(dataloader):
    print(list(sampler))

# for idx, data in enumerate(dataloader):
#     images, target = data
#     print(target)
#     break