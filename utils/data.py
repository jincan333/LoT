import hydra
from torch.utils.data import random_split
import torchvision
import torch
import math
from upcycle import cuda
import copy
from torch.utils.data import TensorDataset, DataLoader
import random
import os

from torchvision.datasets.folder import ImageFolder

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def get_loaders(config):
    train_transform, test_transform = get_augmentation(config)
    if config.dataset.name == 'tiny_imagenet':
        train_dataset = ImageFolder(
            root=os.path.join(hydra.utils.get_original_cwd(), config.dataset.root_dir, 'train'),
            transform=train_transform
        )
        test_dataset = ImageFolder(
            root=os.path.join(hydra.utils.get_original_cwd(), config.dataset.root_dir, 'val'),
            transform=test_transform
        )
    else:
        config.dataset.init.root = os.path.join(hydra.utils.get_original_cwd(), config.dataset.init.root)
        train_dataset = hydra.utils.instantiate(config.dataset.init, train=True, transform=train_transform)
        test_dataset = hydra.utils.instantiate(config.dataset.init, train=False, transform=test_transform)

    if config.dataset.shuffle_train_targets.enabled:
        random.seed(config.dataset.shuffle_train_targets.seed)
        num_shuffled = int(len(train_dataset) * config.dataset.shuffle_train_targets.ratio)
        shuffle_start = random.randint(0, len(train_dataset) - num_shuffled)
        target_copy = train_dataset.targets[shuffle_start:shuffle_start + num_shuffled]
        random.seed(config.dataset.shuffle_train_targets.seed)  # for backwards-compatibility
        random.shuffle(target_copy)
        train_dataset.targets[shuffle_start:shuffle_start + num_shuffled] = target_copy

    subsample_ratio = config.dataset.subsample.ratio
    if subsample_ratio < 1.0:
        train_splits = split_dataset(train_dataset, subsample_ratio,
                                     config.dataset.subsample.seed)
        train_dataset = train_splits[config.dataset.subsample.split]
    else:
        train_splits = [train_dataset]
    if config.trainer.eval_dataset == 'val':
        train_dataset, test_dataset = split_dataset(train_dataset, 0.8)

    train_loader = hydra.utils.instantiate(config.dataloader, dataset=train_dataset)
    test_loader = hydra.utils.instantiate(config.dataloader, dataset=test_dataset)

    return train_loader, test_loader, train_splits


def split_dataset(dataset, ratio, seed=None):
    num_total = len(dataset)
    num_split = int(num_total * ratio)
    gen = torch.Generator() if seed is None else torch.Generator().manual_seed(seed)
    return random_split(dataset, [num_split, num_total - num_split], gen)


def get_augmentation(config):
    assert 'augmentation' in config.keys()
    transforms_list = []
    if config.augmentation.transforms_list is None:
        pass
    elif len(config.augmentation.transforms_list) > 0:
        transforms_list = [hydra.utils.instantiate(config.augmentation[name])
                           for name in config.augmentation["transforms_list"]]
        if 'random_apply' in config.augmentation.keys() and config.augmentation.random_apply.p < 1:
            transforms_list = [
                hydra.utils.instantiate(config.augmentation.random_apply, transforms=transforms_list)]

    normalize_transforms = [
        torchvision.transforms.ToTensor(),
    ]
    if config.augmentation.normalization == 'zscore':
        # mean subtract and scale to unit variance
        normalize_transforms.append(
            torchvision.transforms.Normalize(config.dataset.statistics.mean_statistics,
                                             config.dataset.statistics.std_statistics)
        )
    elif config.augmentation.normalization == 'unitcube':
        # rescale values to [-1, 1]
        min_vals = config.dataset.statistics.min
        max_vals = config.dataset.statistics.max
        offset = [0.5 * (min_val + max_val) for min_val, max_val in zip(min_vals, max_vals)]
        scale = [(max_val - min_val) / 2 for max_val, min_val in zip(max_vals, min_vals)]
        normalize_transforms.append(
            torchvision.transforms.Normalize(offset, scale)
        )

    train_transform = torchvision.transforms.Compose(transforms_list + normalize_transforms)
    test_transform = torchvision.transforms.Compose(normalize_transforms)
    return train_transform, test_transform


def get_distill_loaders(config, train_loader, synth_data):
    num_real = len(train_loader.dataset)
    num_synth = 0 if synth_data is None else synth_data[0].size(0)
    real_ratio = num_real / (num_real + num_synth)
    real_batch_size = math.ceil(real_ratio * config.dataloader.batch_size)
    synth_batch_size = config.dataloader.batch_size - real_batch_size
    train_loader = DataLoader(train_loader.dataset, shuffle=True, batch_size=real_batch_size)
    if num_synth == 0:
        return train_loader, None
    synth_loader = DataLoader(TensorDataset(*synth_data), shuffle=True, batch_size=synth_batch_size)
    return train_loader, synth_loader

