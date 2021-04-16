from DataLoader import dataloader

data_dir = 'train'
csv_file='sampleSubmission.csv'
dataloader(data_dir,csv_file)
for i in range(epochs):
    for j in range(64):
        images, labels = next(iterator)
        # forward propagation
