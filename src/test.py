from DataLoader import dataloader_training , dataloader_testing
from architecture import Net
from torch import nn
import torch
import os
from torch import optim
from src.architecture import Net
from src.DataLoader import dataloader_testing
from torch.utils.tensorboard import SummaryWriter

def test(args):
    writer = SummaryWriter(args.tensorboard_dir)
    dataloader=dataloader_testing(args.test_dir)
    criterion = nn.CrossEntropyLoss()
    net = Net()
    cost=0
    print("start test")
    global_step = 0
    if os.path.exists(os.path.join(args.model_dir+"model.pt")):
        checkpoint = torch.load(os.path.join(args.model_dir,"model.pt"))
        net.load_state_dict(checkpoint['model_state_dict'])

    for j , (images,labels) in enumerate(dataloader):
        output=net(images)
        loss=criterion(output,labels)
        cost+=loss
        print(rf"loss at {j} iteration : {loss} , cost : {cost/(j+1)}")
        global_step+=1
        writer.add_scalar('Loss', loss, global_step)
        writer.add_scalar('Cost', cost/(j+1), global_step)
