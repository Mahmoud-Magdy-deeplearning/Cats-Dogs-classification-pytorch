from DataLoader import dataloader_training , dataloader_testing
from architecture import Net
from torch import nn
from torch import optim
from src.architecture import Net
from src.DataLoader import dataloader_training


train_data_dir = 'train'
csv_file='sampleSubmission.csv'
iterator_train , len=dataloader_training(train_data_dir,csv_file)
epoches = 300
criterion = nn.NLLLoss()
net = Net()
loss=0
cost=0
optimizer = optim.Adam(net.parameters(), lr=0.01)

for i in range(epochs):
    cost /=len
    print(i+"epoch, loss is"+cost+"\n")
    cost=0
    for j in range(64):

        images, labels = next(iterator)
        # forward propagation
        output=net(images)
        loss=criterion(output,labels)
        optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        optimizer.step()
        cost+=loss