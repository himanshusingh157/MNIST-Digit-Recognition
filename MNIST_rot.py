#Importing some required libraries
import scipy.io as sio
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR

#Importing data
dirr=os.getcwd()
filename=dirr+"/MNIST/mnist-rot_training_data.mat"
X_train=sio.loadmat(filename)['train_data']
filename=dirr+"/MNIST/mnist-rot_training_label.mat"
y_train=sio.loadmat(filename)['train_label'].reshape(-1,1)
filename=dirr+"/MNIST/mnist-rot_test_data.mat"
X_test=sio.loadmat(filename)['test_data']
filename=dirr+"/MNIST/mnist-rot_test_label.mat"
y_test=sio.loadmat(filename)['test_label'].reshape(-1,1)

#Shuffling training data
shuffle=np.random.permutation(X_train.shape[0])
X_train=X_train[shuffle]
y_train=y_train[shuffle]

#Chnaging data to tensors
X_train=torch.from_numpy(X_train)
y_train=torch.from_numpy(y_train)
X_test=torch.from_numpy(X_test)
y_test=torch.from_numpy(y_test)

#Model
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(2,2)
        self.conv1=nn.Conv2d(1,8,3)
        self.conv2=nn.Conv2d(8,32,3,padding=1)
        self.fc1 = nn.Linear(32*6*6,512)
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64,10)
        
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
        output=self.fc3(output)
        return output

#Preparing model and data
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Network().to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
criterion=nn.CrossEntropyLoss()

train_ds=TensorDataset(X_train,y_train)
test_ds=TensorDataset(X_test,y_test)
train_loader=DataLoader(train_ds, batch_size=128)
test_loader=DataLoader(test_ds, batch_size=32)

#Training model
epochs=100
plot_values=[]
for epoch in range(epochs):
    loss=0
    for batch,y in train_loader:
        batch=batch.to(device)
        y=y.view(y.shape[0]).long()
        y=y.to(device)
        optimizer.zero_grad()
        output=model(batch.float())
        train_loss=criterion(output,y)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    scheduler.step()
    loss = loss / len(train_loader)
    if (epoch+1)%5==0:
        print(f'Epoch {epoch+1}/{epochs} : Training Error loss = {loss}')
    plot_values.append(loss)

#plotting result
plt.plot(np.arange(1,epochs+1),plot_values,label='Training error')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.show()

#Testing accuracy of model
loss=0
bs=0
acc=0
with torch.no_grad():
    for batch,y in test_loader:
        batch=batch.to(device)
        y=y.view(y.shape[0]).long()
        y=y.to(device)
        bs+=y.shape[0]
        output=model(batch.float())
        test_loss=criterion(output,y)
        loss+=test_loss
        _,preds=torch.max(output,dim=1)
        acc+=torch.sum(preds==y).item()
loss=loss/len(test_loader)
print(f'Loss on Test data = {loss}')
print(f'Accuracy on test data = {acc/bs}')
