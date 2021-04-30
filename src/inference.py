from os import listdir
from os.path import isfile, join
from src.architecture import Net
from torchvision import datasets, transforms, models

path ="inference/process"
net=Net()
for root, dirs, files in os.walk(path):
    for file in files:
        transform = transforms.Compose([transforms.RandomRotation(30),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])


        image = Image.open(file)
        frame = numpy.asarray(image)
        image = self.transform(frame)
        output=net(images)
        print("the output of file"+file+"is"+output+"\n")
