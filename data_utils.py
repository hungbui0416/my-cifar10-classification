import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader

def get_train_val_loaders(args):
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    train_dataset = datasets.CIFAR10(root='./data', 
                                     train=True, 
                                     download=True, 
                                     transform=train_transform)

    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size,
                              shuffle=True, 
                              num_workers=args.num_workers, 
                              pin_memory=True)

    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    return train_loader, val_loader

def get_test_loader(args):
    test_transform = transforms.Compose([transforms.ToTensor(), 
                                         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    
    test_set = datasets.CIFAR10(root='./data', 
                                train=False, 
                                download=True, 
                                transform=test_transform)

    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return test_loader, test_set.classes