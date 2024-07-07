import numpy as np # type: ignore
import torch # type: ignore
from torch.utils.data.sampler import SubsetRandomSampler # type: ignore
from torchvision import datasets, transforms # type: ignore library


class DataLoader:
    def __init__(self, Cifar_type,data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True):
        self.Cifar_type = Cifar_type
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augment = augment
        self.random_seed = random_seed
        self.valid_size = valid_size
        self.shuffle = shuffle

        self.normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        self.train_transform = self._get_train_transform()
        self.valid_transform = self._get_valid_transform()
        self.test_transform = self._get_test_transform()

        self.train_loader, self.valid_loader = self._get_train_valid_loader()
        self.test_loader = self._get_test_loader()

    def _get_train_transform(self):
        if self.augment:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                self.normalize,
            ])

    def _get_valid_transform(self):
        return transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            self.normalize,
        ])

    def _get_test_transform(self):
        return transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            self.normalize,
        ])

    def _get_train_valid_loader(self):
        if self.Cifar_type =="10":
            train_dataset = datasets.CIFAR10(
                root=self.data_dir, train=True,
                download=True, transform=self.train_transform,
            )
        elif self.Cifar_type =="100":
            train_dataset = datasets.CIFAR100(
                root=self.data_dir, train=True,
                download=True, transform=self.train_transform,
            )
        else:
            print("Selection is out of range")

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        train_indices = indices[split:]
        valid_indices = indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        train_dataset_subset = torch.utils.data.Subset(train_dataset, train_indices)
        valid_dataset_subset = torch.utils.data.Subset(train_dataset, valid_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset_subset, batch_size=self.batch_size, shuffle=self.shuffle
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset_subset, batch_size=self.batch_size, shuffle=self.shuffle
        )

        return train_loader, valid_loader

    def _get_test_loader(self):
        if self.Cifar_type =="10":
            dataset = datasets.CIFAR10(
            root=self.data_dir, train=False,
            download=True, transform=self.test_transform,
        )
        elif self.Cifar_type =="100": 
            dataset = datasets.CIFAR100(
            root=self.data_dir, train=False,
            download=True, transform=self.test_transform,
        )
        else:
             print("Selection is out of range")


        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

        return data_loader