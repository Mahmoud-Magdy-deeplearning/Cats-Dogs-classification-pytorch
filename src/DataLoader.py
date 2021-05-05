import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import sys
import os.path
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
from PIL import Image


class Dataset(Dataset):

    # Constructor
    def __init__(self,data_dir):
        # Image directory
        self.data_dir = data_dir

        # The transform is goint to be used on image

        self.transform = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        # files in the directory
        self.files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]


    def __len__(self):

        return len(self.files)

    # Getter
    def __getitem__(self, idx):
        # The class label for the image

        label = 0 if self.files[idx].split(".")[0] == "dog" else 1

        img_name = join(self.data_dir , self.files[idx])

        # Open image file
        image = Image.open(img_name)


        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)


        return image, label




def dataloader_training (data_dir):
    train_dataset = Dataset(data_dir=data_dir)

    sampler=torch.utils.data.RandomSampler(train_dataset)
    sampler=torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=64, drop_last=True)
    dataloader = DataLoader(train_dataset,batch_sampler=sampler)
    return dataloader


def dataloader_testing (data_dir):
    test_dataset = Dataset(data_dir=data_dir)

    sampler=torch.utils.data.RandomSampler(test_dataset)
    sampler=torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=64, drop_last=True)
    dataloader = DataLoader(test_dataset,batch_sampler=sampler)
    return dataloader
