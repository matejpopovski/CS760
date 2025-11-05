import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Data
tt = transforms.ToTensor()
train = datasets.MNIST('./data', train=True, transform=tt, download=True)
test  = datasets.MNIST('./data', train=False, transform=tt, download=True)
train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test, batch_size=1)

# Model
class Net(nn.Module):
    def __init__(self, d=784, h=128, k=10):
        super().__init__()
        self.W1 = nn.Linear(d, h)
        self.W2 = nn.Linear(h, k)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.view(-1, 784)
        h = self.sigmoid(self.W1(x))
        out = self.softmax(self.W2(h))
        return out

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

# Training
for epoch in range(5):
    loss_sum, correct = 0, 0
    for x, y in train_loader:
        optimizer.zero_grad()
        y_hat = net(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        correct += (y_hat.argmax(1) == y).sum().item()
    print(f"Epoch {epoch+1}: loss={loss_sum/len(train_loader):.3f}, acc={correct/len(train_loader.dataset)*100:.2f}%")

# Testing
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        y_hat = net(x)
        correct += (y_hat.argmax(1) == y).sum().item()
print(f"Test accuracy: {correct/len(test_loader.dataset)*100:.2f}%")
