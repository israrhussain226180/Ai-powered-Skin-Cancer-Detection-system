import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    """
    Creates DataLoaders for train, val, and test sets.
    Assumes data_dir has 'train', 'val', and 'test' subdirectories.
    """
    
    # Standard normalization for ImageNet-based models (like EfficientNet)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, 
                       shuffle=(x == 'train'), num_workers=0, pin_memory=True)
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

if __name__ == "__main__":
    # Test loader creation (assuming data exists)
    try:
        loaders, sizes, classes = get_data_loaders("../../data/splits")
        print(f"Classes: {classes}")
        print(f"Dataset sizes: {sizes}")
    except Exception as e:
        print(f"Could not load data: {e}")
