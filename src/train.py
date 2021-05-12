from DataLoader import dataloader_training , dataloader_testing
from architecture import Net
from torch import nn
import torch
import os
from torch import optim
from src.architecture import Net
from src.DataLoader import dataloader_training
from torch.utils.tensorboard import SummaryWriter

def train(args):
    writer = SummaryWriter(args.tensorboard_dir)
    dataloader=dataloader_training(args.train_dir)
    epochs = 300
    criterion = nn.CrossEntropyLoss()

    net = Net()
    if args.cuda:
        net=net.cuda()
    cost=0
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    print("start train")
    global_step = 0
    start_epoch=0
    if os.path.exists(os.path.join(args.model_dir,"model.pt")):
        checkpoint = torch.load(os.path.join(args.model_dir,"model.pt"))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        global_step = checkpoint['global_step']

    for epoch in range(start_epoch, epochs):
        cost /= len(dataloader)
        print( rf"{epoch+1} epoch")
        cost=0

        for j , (images,labels) in enumerate(dataloader):

            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            output=net(images)
            loss=criterion(output,labels)
            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()
            cost+=loss
            print(rf"loss at {j} iteration : {loss} , cost : {cost/(j+1)}")
            global_step+=1
            writer.add_scalar('Loss', loss, global_step)
            writer.add_scalar('Cost', cost/(j+1), global_step)

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, os.path.join(args.model_dir,"model.pt"))