from os import listdir
from os.path import isfile, join
from src.architecture import Net
from torchvision import datasets, transforms, models
from src.env import inference_path
from PIL import Image


path = inference_path()
net=Net()
transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

for root, dirs, files in os.walk(path):
    for file in files:


        image = Image.open(file)
        image = self.transform(frame)
        output=net(images)
        if output == 0:  print("the output of file"+file+"is Cat"+"\n")
        else:  print("the output of file"+file+"is Dog"+"\n")

