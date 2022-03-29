#%%
import numpy as np
import pickle

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

dpath = "./data"
PATH = 'saved/cifar_net.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#%%
def load_mnist():
    with open(f"{dpath}/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

X_train, y_train, X_test, y_test = load_mnist()


#%%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        logits = self.ffn(x)
        return logits

net = Net()
print(net)

#%%
def load_mnist():
    with open(f"{dpath}/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

X_train, y_train, X_test, y_test = load_mnist()

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.label[idx]
        return image, label

#%%
X_train = np.array(X_train, dtype=np.float32)
y_train = torch.from_numpy(y_train)
train_dataset = CustomImageDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=.001, momentum=0.9)

net = net.to(device)
for epoch in range(2):
    running_loss = 0.0
    for i, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f}')
print('Finished Training')

#%%
PATH = 'saved/cifar_net.pth'
torch.save(net.state_dict(), PATH)

# %%

net = Net()
net.load_state_dict(torch.load(PATH))


X_test = np.array(X_test, dtype=np.float32)
# y_test = torch.from_numpy(y_test)
test_dataset = CustomImageDataset(X_test, y_test)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

correct = 0
total = 0
#%%
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# %%
