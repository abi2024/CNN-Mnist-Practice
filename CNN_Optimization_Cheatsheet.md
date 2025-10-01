# 🧠 PyTorch CNN Optimization Cheatsheet

A comprehensive guide for building and optimizing Convolutional Neural Networks in PyTorch.

---

## 📐 Core Architecture Components

### **Convolutional Layers**

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

**Parameters:**
- `in_channels` — Input feature maps
- `out_channels` — Number of filters (output channels)
- `kernel_size` — Filter size (e.g., 3 for 3×3)
- `stride` — Step size for convolution
- `padding` — Zero-padding around input

**Output Size Formula:**
```
Output = (W - K + 2P) / S + 1
```
Where: W=input width, K=kernel size, P=padding, S=stride

**Example:**
```python
self.conv1 = nn.Conv2d(1, 10, kernel_size=3)  # 28×28 → 26×26
```

---

### **Batch Normalization**

```python
nn.BatchNorm2d(num_features)
```

- `num_features` = number of channels from previous Conv layer
- **Benefits:** Stabilizes training, allows higher learning rates
- **Placement:** After Conv, before activation

```python
self.bn1 = nn.BatchNorm2d(10)  # For 10 output channels
```

---

### **Activation Functions**

```python
F.relu(x)                    # Rectified Linear Unit (most common)
F.leaky_relu(x)             # Prevents dying ReLU problem
F.log_softmax(x, dim=-1)    # For final output layer
```

**Common Pattern:**
```python
x = F.relu(self.bn1(self.conv1(x)))
```

---

### **Pooling Layers**

```python
nn.MaxPool2d(kernel_size, stride)           # Downsampling
nn.AvgPool2d(kernel_size, stride)           # Average pooling
nn.AdaptiveAvgPool2d((H, W))                # Adaptive to specific size
```

**Effect:** Reduces spatial dimensions (typically by 2x)

```python
self.pool = nn.MaxPool2d(2, 2)  # 26×26 → 13×13
```

---

### **Regularization**

```python
nn.Dropout2d(p=0.25)    # Spatial dropout (drops entire channels)
nn.Dropout(p=0.5)       # Standard dropout for FC layers
```

**When to use:**
- Between convolutional blocks
- Before fully connected layers
- Prevents overfitting

---

### **Fully Connected Layers**

```python
nn.Linear(in_features, out_features)
```

**Calculate `in_features`:**
```python
in_features = channels × height × width
# After all conv/pool operations
```

**Example:**
```python
self.fc1 = nn.Linear(20 * 12 * 12, 10)  # 2880 → 10 classes
```

---

## 🔄 Forward Pass Patterns

### **Basic Conv Block**
```python
x = F.relu(self.bn1(self.conv1(x)))
```

### **Conv Block with Pooling**
```python
x = self.pool(F.relu(self.bn2(self.conv2(x))))
```

### **Complete Forward Pass**
```python
def forward(self, x):
    # Convolutional blocks
    x = F.relu(self.bn1(self.conv1(x)))
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    x = self.dropout(x)

    # Flatten for fully connected layer
    x = x.view(-1, channels * height * width)  # or x.flatten(1)

    # Classification layer
    x = self.fc1(x)
    return F.log_softmax(x, dim=-1)
```

---

## ⚡ Parameter Reduction Techniques

### **1. Reduce Channel Count**
```python
# Before: 1,820 params
Conv2d(10, 20, kernel_size=3)

# After: ~780 params (60% reduction)
Conv2d(8, 16, kernel_size=3)
```

### **2. Use 1×1 Convolutions**
Reduce channels before expensive 3×3 convolutions:
```python
nn.Conv2d(64, 16, kernel_size=1)  # Bottleneck
nn.Conv2d(16, 64, kernel_size=3)  # Expensive operation
```

### **3. Depthwise Separable Convolutions**
Replace standard convolution with depthwise + pointwise:
```python
# Standard Conv
nn.Conv2d(32, 64, kernel_size=3)  # 18,432 params

# Separable Conv (same output)
nn.Conv2d(32, 32, kernel_size=3, groups=32)  # Depthwise: 288 params
nn.Conv2d(32, 64, kernel_size=1)             # Pointwise: 2,048 params
# Total: 2,336 params (87% reduction!)
```

### **4. Global Average Pooling**
Replace large FC layers:
```python
# Before
nn.Linear(20 * 12 * 12, 10)  # 28,810 params

# After
nn.AdaptiveAvgPool2d((1, 1))  # Reduces to 20×1×1
nn.Linear(20, 10)              # Only 210 params!
```

### **5. Smaller Kernel Sizes**
```python
nn.Conv2d(10, 20, kernel_size=3)  # 9 params per filter
nn.Conv2d(10, 20, kernel_size=5)  # 25 params per filter
```

---

## 🎯 Training Optimization

### **Optimizers**

```python
# Adam - Adaptive learning rate, good default
optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

# SGD - Often achieves better final accuracy
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# AdamW - Adam with better weight decay
optim.AdamW(model.parameters(), lr=0.001)
```

**Choosing Learning Rate:**
- Too high: Training unstable, loss oscillates
- Too low: Training too slow, gets stuck
- Sweet spot: 0.001 - 0.01 for Adam, 0.01 - 0.1 for SGD

---

### **Learning Rate Schedulers**

```python
# Step decay every N epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Reduce on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.1
)

# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# One Cycle (for faster training)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, epochs=10, steps_per_epoch=len(train_loader)
)
```

**Usage:**
```python
for epoch in range(num_epochs):
    train(...)
    test(...)
    scheduler.step()  # Update learning rate
```

---

### **Loss Functions**

```python
# Classification (most common)
nn.CrossEntropyLoss()  # Use with raw logits (no softmax)
nn.NLLLoss()           # Use with log_softmax output

# With class weights (for imbalanced data)
weights = torch.tensor([0.5, 2.0, 1.0, ...])
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## 🖼️ Data Augmentation

### **Training Transforms**
```python
train_transforms = transforms.Compose([
    transforms.RandomRotation((-15, 15)),          # Rotation invariance
    transforms.RandomApply([
        transforms.CenterCrop(22)
    ], p=0.1),                                      # Random cropping
    transforms.Resize((28, 28)),                    # Resize back
    transforms.ColorJitter(brightness=0.2),         # Brightness variation
    transforms.RandomHorizontalFlip(p=0.5),         # Horizontal flip
    transforms.ToTensor(),                          # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))     # Normalize (MNIST)
])
```

### **Test Transforms**
```python
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

**Common Augmentations:**
| Transform | Use Case |
|-----------|----------|
| `RandomRotation` | Rotation invariance |
| `RandomCrop` | Scale invariance |
| `RandomHorizontalFlip` | Left-right symmetry |
| `ColorJitter` | Lighting variations |
| `RandomAffine` | Translation/shear |

---

## 📊 DataLoader Configuration

```python
kwargs = {
    'batch_size': 128,      # Larger = faster, needs more memory
    'shuffle': True,        # Always True for training
    'num_workers': 2,       # Parallel data loading (2-4 optimal)
    'pin_memory': True      # Faster GPU transfer if CUDA
}

train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
```

**Batch Size Guidelines:**
- Small (16-32): Less memory, noisier gradients, better generalization
- Medium (64-128): Good balance
- Large (256-512): Faster training, needs learning rate adjustment

---

## 📏 Quick Reference: Common Operations

### **Output Size Calculations**

| Operation | Formula | Example |
|-----------|---------|---------|
| **Conv2d** | `(W - K + 2P) / S + 1` | 28→26 (K=3, P=0, S=1) |
| **MaxPool2d** | `W / stride` | 26→13 (stride=2) |
| **Same padding** | `P = (K-1)/2` | K=3 → P=1 (28→28) |

### **Parameter Count**

| Layer | Formula | Example |
|-------|---------|---------|
| **Conv2d** | `K² × C_in × C_out + C_out` | 3²×10×20+20 = 1,820 |
| **Linear** | `in × out + out` | 2880×10+10 = 28,810 |
| **BatchNorm2d** | `2 × num_features` | 2×10 = 20 |

### **Memory Usage (Approximate)**

| Component | Formula |
|-----------|---------|
| **Activation maps** | `batch × channels × H × W × 4 bytes` |
| **Parameters** | `total_params × 4 bytes` |
| **Gradients** | `total_params × 4 bytes` |

---

## 🔧 Model Analysis Tools

### **Count Parameters**
```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,} | Trainable: {trainable_params:,}")
```

### **Layer-wise Breakdown**
```python
for name, param in model.named_parameters():
    print(f"{name:20} | Shape: {str(list(param.shape)):20} | Params: {param.numel():,}")
```

### **Model Summary**
```python
from torchsummary import summary
summary(model, input_size=(1, 28, 28))
```

### **Check GPU**
```python
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

---

## 🏋️ Training Loop Template

### **Training Function**
```python
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # Set to training mode
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to device
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate averages
    train_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)

    return accuracy, train_loss
```

### **Testing Function**
```python
def test(model, device, test_loader, criterion):
    model.eval()  # Set to evaluation mode
    test_loss = 0
    correct = 0

    with torch.no_grad():  # Disable gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy, test_loss
```

### **Main Training Loop**
```python
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, num_epochs + 1):
    train_acc, train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
    test_acc, test_loss = test(model, device, test_loader, criterion)
    scheduler.step()

    print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
```

---

## 🎨 Best Practices & Patterns

### **Standard Architecture Order**
```
Input → Conv → BatchNorm → ReLU → Pool → Dropout → ... → Flatten → FC → Output
```

### **Common Block Patterns**

**Basic Block:**
```python
x = F.relu(self.bn1(self.conv1(x)))
```

**Residual Block:**
```python
identity = x
x = F.relu(self.bn1(self.conv1(x)))
x = self.bn2(self.conv2(x))
x += identity  # Skip connection
x = F.relu(x)
```

**Inception-style Block:**
```python
# Multiple parallel paths
path1 = self.conv1x1(x)
path2 = self.conv3x3(x)
path3 = self.conv5x5(x)
path4 = self.maxpool(x)
x = torch.cat([path1, path2, path3, path4], dim=1)
```

### **Device Management**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data, target = data.to(device), target.to(device)
```

### **Saving & Loading**
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = Net()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

---

## 🐛 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Out of memory** | Batch size too large | Reduce batch size or model size |
| **Slow training** | CPU instead of GPU | Check `device`, move data/model to GPU |
| **Loss not decreasing** | Learning rate too low/high | Adjust LR or try different optimizer |
| **Overfitting** | Model too complex | Add dropout, data augmentation, reduce capacity |
| **Underfitting** | Model too simple | Increase capacity, train longer |
| **NaN loss** | Learning rate too high | Reduce learning rate, check data normalization |
| **Size mismatch** | Wrong flattened size | Calculate `C × H × W` after conv/pool |

---

## 📚 Optimization Checklist

- [ ] **Data Augmentation** — Improves generalization
- [ ] **Batch Normalization** — Stabilizes training
- [ ] **Dropout** — Prevents overfitting
- [ ] **Learning Rate Schedule** — Better convergence
- [ ] **Proper Initialization** — PyTorch does this by default
- [ ] **GPU Utilization** — Move model and data to CUDA
- [ ] **Batch Size Tuning** — Balance speed and memory
- [ ] **Early Stopping** — Prevent overfitting
- [ ] **Gradient Clipping** — Prevent exploding gradients
- [ ] **Mixed Precision** — Faster training with `torch.cuda.amp`

---

## 🚀 Advanced Techniques

### **Mixed Precision Training**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### **Gradient Accumulation**
```python
accumulation_steps = 4

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### **Transfer Learning**
```python
# Load pretrained model
model = torchvision.models.resnet18(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(512, num_classes)

# Train only final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

---

## 📖 Resources

- **Official Docs:** [pytorch.org/docs](https://pytorch.org/docs)
- **Tutorials:** [pytorch.org/tutorials](https://pytorch.org/tutorials)
- **Vision Models:** [pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
- **Papers with Code:** [paperswithcode.com](https://paperswithcode.com)

---

**Created from:** `Lightweight_CNN.ipynb`
**Last Updated:** 2025-10-01
