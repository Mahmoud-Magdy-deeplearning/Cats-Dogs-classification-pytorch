from DataLoader import dataloader_training , dataloader_testing
from architecture import Net
from torch import nn
import torch
import os
from torch import optim
from src.architecture import Net
from src.DataLoader import dataloader_training
from src.env import train_path , model_path , tensorboard_path
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(tensorboard_path())

train_data_dir = train_path()
dataloader=dataloader_training(train_data_dir)
epochs = 300
criterion = nn.CrossEntropyLoss()
net = Net()
loss=0
cost=0
optimizer = optim.Adam(net.parameters(), lr=0.01)
print("start train")
global_step = 0
start_epoch=0
if os.path.exists(os.path.join(model_path(),"model.pt")):
    checkpoint = torch.load(os.path.join(model_path(),"model.pt"))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']+1
    global_step = checkpoint['global_step']

for epoch in range(start_epoch, epochs):
    cost /= len(dataloader)
    print( rf"{epoch+1} epoch")
    cost=0

    for j , (images,labels) in enumerate(dataloader):
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
    }, os.path.join(model_path(),"model.pt"))