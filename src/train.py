from DataLoader import dataloader_training , dataloader_testing
from architecture import Net
from torch import nn
from torch import optim


test_data_dir = 'test'
train_data_dir = 'train'
csv_file='sampleSubmission.csv'
iterator_train=dataloader_training(train_data_dir,csv_file)
iterator_test=dataloader_testing(test_data_dir,csv_file)
epoches = 300
optimizer = optim.Adam(Net.parameters(), lr=0.01)
criterion = nn.NLLLoss()

for i in range(epochs):
    for j in range(64):
        images, labels = next(iterator)
        # forward propagation
        output=Net.forward(images)
        optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        optimizer.step()
