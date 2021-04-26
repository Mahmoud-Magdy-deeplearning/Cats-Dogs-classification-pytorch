import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import sys
import os.path
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        # Image directory
        self.data_dir = data_dir

        # The transform is goint to be used on image

        self.transform = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        # Load the CSV file contians image info
        self.data_name = pd.read_csv(csv_file)

        # Number of images in dataset
        self.len = self.data_name.shape[0]

        # Get the length

    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):
        # Image file path
        img_name = self.data_dir + self.data_name.iloc[idx, 2]

        # Open image file
        image = Image.open(img_name)

        # The class label for the image
        y = self.data_name.iloc[idx, 3]

        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y




def dataloader_training (self, data_dir,csv_file):
    train_dataset = Dataset(csv_file=csv_file
                                , data_dir=data_dir)
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    iterator = iter(dataloader)
    return iterator


def dataloader_testing (self, data_dir,csv_file):
    test_dataset = Dataset(csv_file=csv_file
                                , data_dir=data_dir)
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    iterator = iter(dataloader)
    return iterator