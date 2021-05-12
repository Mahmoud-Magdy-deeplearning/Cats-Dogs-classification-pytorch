from os import listdir
from os.path import isfile, join
from src.architecture import Net
from torchvision import datasets, transforms, models
from PIL import Image


def inference(args):
    path = args.inf_dir

    net=Net()
    if os.path.exists(os.path.join(args.model_dir+"model.pt")):
        checkpoint = torch.load(os.path.join(args.model_dir,"model.pt"))
        net.load_state_dict(checkpoint['model_state_dict'])

    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    for root, dirs, files in os.walk(path):
        for file in files:


            image = Image.open(file)
            image = transform(image)
            output=net(image)
            if output == 0:  print("the output of file"+file+"is Cat"+"\n")
            else:  print("the output of file"+file+"is Dog"+"\n")

