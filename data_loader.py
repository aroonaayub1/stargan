from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images



class CompCars(data.Dataset):
    """Dataset class for the CompCars dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CompCars dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CompCars attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i # make dictionaries for attribute to indexes
            self.idx2attr[i] = attr_name # make dictionaries for indexes to attribute 

        lines = lines[2:]
        random.seed(1234)                   # Initialize random numbers from specified seed - same random number generation - Same random list of images
        random.shuffle(lines)               # Shuffle all lines/images
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:] # list of values against each label either 1 or -1

            label = []
            for attr_name in self.selected_attrs:               # Append all attribute names in label
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 13000:
                self.test_dataset.append([filename, label])     #Append first 2000 random images to test dataset
            else:   
                self.train_dataset.append([filename, label])    #Append 2000+ images to training dataset

        print('Finished preprocessing the CompCars dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label), filename

    def __len__(self):
        """Return the number of images."""
        return self.num_images




def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip()) # Randomly flips the image
    # transform.append(T.CenterCrop(crop_size)) # crops from center 
    transform.append(T.Resize(image_size)) # resizes the cropped image
    transform.append(T.ToTensor()) # converts to tensor - numpy.ndarray (H x W x C) -> torch.FloatTensor of shape (C x H x W)
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))) # normalizes the image (by using mean and standard deviation)
    transform = T.Compose(transform)

    # Load images and attributes
    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)
    elif dataset == 'CompCars':
        dataset = CompCars(image_dir, attr_path, selected_attrs, transform, mode)

    # Builtin function - Divides the images and attributes into batches according to batch size
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    
    # Returns loaded images and attributes in batches
    return data_loader