from DataLoader import dataloader_training , dataloader_testing
from architecture import Net
from torch import nn
from torch import optim
from src.architecture import Net
from src.DataLoader import dataloader_training



test_data_dir = 'test'
csv_file='sampleSubmission.csv'
iterator_test=dataloader_testing(test_data_dir,csv_file)
criterion = nn.NLLLoss()
net=Net()
for j in range(64):
    images, labels = next(iterator)
    # forward propagation
    output=net(images)
    loss=criterion(output,labels)
    loss += criterion

loss /=len

print(j + ", loss is" + loss + "\n")
