from torch.utils.data import DataLoader
import os
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import  transforms

def image_transform(args):
    if args.dataset=='cifar100':
        mean_statistics = (0.5071, 0.4867, 0.4408)
        std_statistics = (0.2675, 0.2565, 0.2761)
        max_values = (1.0, 1.0, 1.0)
        min_values = (0.0, 0.0, 0.0)
        args.num_classes=100
    elif args.dataset=='cifar10':
        mean_statistics = (0.4914, 0.4822, 0.4465)
        std_statistics = (0.2470, 0.2435, 0.2616)
        max_values = (1.0, 1.0, 1.0)
        min_values = (0.0, 0.0, 0.0)
        args.num_classes=100
    offset = [0.5 * (min_val + max_val) for min_val, max_val in zip(min_values, max_values)]
    scale = [(max_val - min_val) / 2 for max_val, min_val in zip(max_values, min_values)]
    normalize = transforms.Normalize(mean=offset, std=scale)
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=args.input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return train_transform, test_transform


def get_torch_dataset(args):
    data_path = os.path.join(args.datadir, args.dataset)
    train_transform, test_transform = image_transform(args)
    if args.dataset == "cifar10":
        train_set = CIFAR10(data_path, train=True, transform=train_transform, download=True)
        test_set = CIFAR10(data_path, train=False, transform=test_transform, download=True)
    elif args.dataset == "cifar100":
        train_set = CIFAR100(data_path, train=True, transform=train_transform, download=True)
        test_set = CIFAR100(data_path, train=False, transform=test_transform, download=True)
    else:
        raise NotImplementedError(f"{args.dataset} not supported")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f'Dataset information: {args.dataset}\t {len(train_set)} images for training \t {len(test_set)} images for testing\t')
    return train_loader, test_loader

