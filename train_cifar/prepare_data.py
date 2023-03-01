import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, RandomSampler
from distributed_utils import is_main_process

def get_loader(args, train, download=False):
    if is_main_process():
        print('==> Preparing data..\n')
    if train:
        if args.aug:
            print("Loading training data with augmentation")
            transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        else:
            print("Loading training data without augmentation")
            transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        dataset = torchvision.datasets.CIFAR10(root=args.data_pth, train=True, download=download, transform=transform_train)


    else:
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        dataset = torchvision.datasets.CIFAR10(root=args.data_pth, train=False, download=download, transform=transform_test)

    if args.replacement and train:
        num_samples = args.batch_size_per_gpu * args.steps_per_epoch
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        if args.debug and is_main_process():
            print(f"num samples per epoch per gpu: {num_samples}")
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=args.seed)
        

    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, sampler=sampler, num_workers=args.nw, drop_last=train)
    # for idx, data in enumerate(dataloader):
    #     images, target = data
    #     print(target)
    #     break


        
    return dataloader, sampler
