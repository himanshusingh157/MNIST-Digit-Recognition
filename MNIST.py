#Importing some required libraries
import scipy.io as sio
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

#Importing data
dirr=os.getcwd()
filename=dirr+"/MNIST/mnist_training_data.mat"
X_train=sio.loadmat(filename)['training_data']
filename=dirr+"/MNIST/mnist_training_label.mat"
y_train=sio.loadmat(filename)['training_label']
filename=dirr+"/MNIST/mnist_test_data.mat"
X_test=sio.loadmat(filename)['test_data']
filename=dirr+"/MNIST/mnist_test_label.mat"
y_test=sio.loadmat(filename)['test_label']

#Shuffling training data
shuffle=np.random.permutation(X_train.shape[0])
X_train=X_train[shuffle]
y_train=y_train[shuffle]

#Chnaging data to tensors
X_train=torch.from_numpy(X_train)
y_train=torch.from_numpy(y_train)
X_test=torch.from_numpy(X_test)
y_test=torch.from_numpy(y_test)

#MODEL
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(2,2)
        self.conv1=nn.Conv2d(1,8,3)
        self.conv2=nn.Conv2d(8,16,3,stride=2)
        self.fc1 = nn.Linear(16*3*3,48)
        self.fc2 = nn.Linear(48,10)
        
    def forward(self,input_feature):
        input_feature=input_feature.view(-1,1,28,28)
        output=self.conv1(input_feature)
        output=self.relu(output)
        output=self.maxpool(output)
        output=self.conv2(output)
        output=self.relu(output)
        output=self.maxpool(output)
        output=output.view(output.shape[0],-1)
        output=self.fc1(output)
        output=self.relu(output)
        output=self.fc2(output)
        return output

#Checking if cuda is available
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#definig model
model = Network().to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()

train_ds=TensorDataset(X_train,y_train)
test_ds=TensorDataset(X_test,y_test)
train_loader=DataLoader(train_ds, batch_size=64)
test_loader=DataLoader(test_ds, batch_size=32)

#Training Model
epochs=50
plot_values=[]
for epoch in range(epochs):
    loss=0
    for batch,y in train_loader:
        batch=batch.to(device)
        y=y.view(y.shape[0]).long()
        optimizer.zero_grad()
        output=model(batch.float())
        train_loss=criterion(output,y)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    loss = loss / len(train_loader)
    plot_values.append(loss)

plt.plot(np.arange(1,epochs+1),plot_values,label='Training error')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.show()

#Testing Model
loss=0
bs=0
acc=0
with torch.no_grad():
    for batch,y in test_loader:
        batch=batch.to(device)
        y=y.view(y.shape[0]).long()
        bs+=y.shape[0]
        output=model(batch.float())
        test_loss=criterion(output,y)
        loss+=test_loss
        _,preds=torch.max(output,dim=1)
        acc+=torch.sum(preds==y).item()
loss=loss/len(test_loader)
print(f'Loss on Test data = {loss}')
print(f'Accuracy on test data = {acc/bs}')

