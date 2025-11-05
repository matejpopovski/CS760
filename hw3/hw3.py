import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# -----------------------
# 1. Data
# -----------------------
# Pixels scaled to [0,1] automatically by ToTensor() (divides by 255)
transform = transforms.ToTensor()

train = datasets.MNIST('./data', train=True, transform=transform, download=True)
test  = datasets.MNIST('./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test, batch_size=1)

# -----------------------
# 2. Model
# -----------------------
class TwoLayerNet(nn.Module):
    def __init__(self, d=784, h=128, k=10):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, k)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # Manual weight initialization: uniform in [-1/sqrt(k), 1/sqrt(k)]
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, -1/math.sqrt(p.size(1)), 1/math.sqrt(p.size(1)))

    def forward(self, x):
        x = x.view(-1, 784)
        h = self.sigmoid(self.fc1(x))
        out = self.softmax(self.fc2(h))
        return out

net = TwoLayerNet()

# -----------------------
# 3. Loss + Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

# -----------------------
# 4. Training
# -----------------------
for epoch in range(5):  # exactly 5 epochs
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
    avg_loss = loss_sum / len(train_loader)
    acc = correct / len(train_loader.dataset) * 100
    print(f"Epoch {epoch+1}: loss={avg_loss:.3f}, acc={acc:.2f}%")

# -----------------------
# 5. Evaluation
# -----------------------
net.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        y_hat = net(x)
        correct += (y_hat.argmax(1) == y).sum().item()

test_acc = correct / len(test_loader.dataset) * 100
print(f"Test accuracy: {test_acc:.2f}%")


# ============================================================
# Exercise 3.4 — Effect of Normalization
# ============================================================
print("\n\n=== Exercise 3.4: Training with Normalization ===")

# 1. Mean and standard deviation for MNIST
mean, std = 0.1307, 0.3081
print(f"Mean pixel value (μ): {mean}")
print(f"Standard deviation (σ): {std}")

# 2. Normalized transform
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# 3. Reload normalized data
train_norm = datasets.MNIST('./data', train=True, transform=transform_norm, download=True)
test_norm  = datasets.MNIST('./data', train=False, transform=transform_norm, download=True)

train_loader_norm = torch.utils.data.DataLoader(train_norm, batch_size=1, shuffle=True)
test_loader_norm  = torch.utils.data.DataLoader(test_norm, batch_size=1)

# 4. Reuse same model and hyperparameters
net_norm = TwoLayerNet()  # same as before (sigmoid, softmax)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_norm.parameters(), lr=1e-3)

# 5. Train for 5 epochs
for epoch in range(5):
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
    avg_loss = loss_sum / len(train_loader_norm)
    acc = correct / len(train_loader_norm.dataset) * 100
    print(f"[Normalized] Epoch {epoch+1}: loss={avg_loss:.3f}, acc={acc:.2f}%")

# 6. Evaluate on test set
net_norm.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader_norm:
        y_hat = net_norm(x)
        correct += (y_hat.argmax(1) == y).sum().item()

test_acc_norm = correct / len(test_loader_norm.dataset) * 100
print(f"[Normalized] Test accuracy: {test_acc_norm:.2f}%")

print("\n=> Normalization centered and scaled pixel intensities, improving gradient conditioning and stabilizing training convergence.")



# ============================================================
# Exercise 3.5 — Activation Functions
# ============================================================
print("\n\n=== Exercise 3.5: Training with ReLU Activation ===")

# Reuse same normalization transform as in 3.4
mean, std = 0.1307, 0.3081
transform_relu = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# Load normalized MNIST dataset again
train_relu = datasets.MNIST('./data', train=True, transform=transform_relu, download=True)
test_relu  = datasets.MNIST('./data', train=False, transform=transform_relu, download=True)

train_loader_relu = torch.utils.data.DataLoader(train_relu, batch_size=1, shuffle=True)
test_loader_relu  = torch.utils.data.DataLoader(test_relu, batch_size=1)

# Define new model with ReLU activation
class TwoLayerNetReLU(nn.Module):
    def __init__(self, d=784, h=128, k=10):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, k)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        # weight init same as before
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, -1/math.sqrt(p.size(1)), 1/math.sqrt(p.size(1)))

    def forward(self, x):
        x = x.view(-1, 784)
        h = self.relu(self.fc1(x))
        out = self.softmax(self.fc2(h))
        return out

# Instantiate and train
net_relu = TwoLayerNetReLU()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net_relu.parameters(), lr=1e-3)

for epoch in range(5):
    net_relu.train()
    loss_sum, correct = 0.0, 0
    for x, y in train_loader_relu:
        optimizer.zero_grad()
        y_hat = net_relu(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        correct += (y_hat.argmax(1) == y).sum().item()
    avg_loss = loss_sum / len(train_loader_relu)
    acc = correct / len(train_loader_relu.dataset) * 100
    print(f"[ReLU] Epoch {epoch+1}: loss={avg_loss:.3f}, acc={acc:.2f}%")

# Evaluate on test set
net_relu.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader_relu:
        y_hat = net_relu(x)
        correct += (y_hat.argmax(1) == y).sum().item()

test_acc_relu = correct / len(test_loader_relu.dataset) * 100
print(f"[ReLU] Test accuracy: {test_acc_relu:.2f}%")

print("\n=> ReLU accelerates convergence and yields higher accuracy by avoiding gradient saturation seen with sigmoid activations.")



# ============================================================
# Exercise 3.6 — Confusion Matrix
# ============================================================
print("\n\n=== Exercise 3.6: Confusion Matrix for ReLU Model ===")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Ensure evaluation mode
net_relu.eval()

# Collect all predictions and labels
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader_relu:
        y_hat = net_relu(x)
        all_preds.append(y_hat.argmax(1).item())
        all_labels.append(y.item())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Confusion Matrix — ReLU Model (MNIST)")
plt.savefig("confusion_matrix_relu.png")
plt.close(fig)
print("Confusion matrix saved as confusion_matrix_relu.png")

# Analyze misclassifications
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
most_confused_idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
mislabel_class = most_confused_idx[0]
confused_as_class = most_confused_idx[1]

print(f"\n(1) Most misclassified true class: {mislabel_class}")
print(f"(2) Model most often confuses {mislabel_class} → {confused_as_class}")




