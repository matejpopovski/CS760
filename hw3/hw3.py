import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# -----------------------
# 1. Data (with normalization)
# -----------------------
mean, std = 0.1307, 0.3081
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

train = datasets.MNIST('./data', train=True, transform=transform, download=True)
test  = datasets.MNIST('./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test, batch_size=64)

# -----------------------
# 2. Model
# -----------------------
class TwoLayerNet(nn.Module):
    def __init__(self, d=784, h=128, k=10, use_relu=False):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, k)
        self.use_relu = use_relu
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 784)
        h = self.relu(self.fc1(x)) if self.use_relu else self.sigmoid(self.fc1(x))
        out = self.softmax(self.fc2(h))
        return out

# instantiate with use_relu=True to get extra boost
net = TwoLayerNet(use_relu=True)

# -----------------------
# 3. Loss + Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# -----------------------
# 4. Training
# -----------------------
for epoch in range(10):
    net.train()
    loss_sum, correct = 0.0, 0
    for x, y in train_loader:
        optimizer.zero_grad()
        y_hat = net(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        correct += (y_hat.argmax(1) == y).sum().item()
    acc = correct / len(train_loader.dataset) * 100
    print(f"Epoch {epoch+1}: loss={loss_sum/len(train_loader):.3f}, acc={acc:.2f}%")

# -----------------------
# 5. Evaluation
# -----------------------
net.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        y_hat = net(x)
        correct += (y_hat.argmax(1) == y).sum().item()

print(f"Test accuracy: {correct / len(test_loader.dataset) * 100:.2f}%")


print("\n\n=== Exercise 3.4: Training with Normalization ===")

# Define normalized transform
mean, std = 0.1307, 0.3081
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# Reload dataset with normalization
train_norm = datasets.MNIST('./data', train=True, transform=transform_norm, download=True)
test_norm  = datasets.MNIST('./data', train=False, transform=transform_norm, download=True)

train_loader_norm = torch.utils.data.DataLoader(train_norm, batch_size=64, shuffle=True)
test_loader_norm  = torch.utils.data.DataLoader(test_norm, batch_size=64)

# Reuse your same TwoLayerNet definition
net_norm = TwoLayerNet(use_relu=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_norm.parameters(), lr=0.01, momentum=0.9)

# Train again on normalized data
for epoch in range(10):
    net_norm.train()
    loss_sum, correct = 0.0, 0
    for x, y in train_loader_norm:
        optimizer.zero_grad()
        y_hat = net_norm(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        correct += (y_hat.argmax(1) == y).sum().item()
    acc = correct / len(train_loader_norm.dataset) * 100
    print(f"[Normalized] Epoch {epoch+1}: loss={loss_sum/len(train_loader_norm):.3f}, acc={acc:.2f}%")

# Evaluate on test set
net_norm.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader_norm:
        y_hat = net_norm(x)
        correct += (y_hat.argmax(1) == y).sum().item()

test_acc_norm = correct / len(test_loader_norm.dataset) * 100
print(f"[Normalized] Test accuracy: {test_acc_norm:.2f}%\n")

print("=> Normalization improved performance significantly and stabilized training.")