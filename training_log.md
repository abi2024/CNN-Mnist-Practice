# MNIST CNN Training Log

## Project Goals
- **Parameter Count**: < 25,000 parameters
- **Test Accuracy**: ≥ 95% in 1 epoch

---

## Iteration 1

### Architecture Details
```
Model: Net
├── conv1: Conv2d(1, 10, kernel_size=3) → 10x26x26
│   └── bn1: BatchNorm2d(10)
├── conv2: Conv2d(10, 20, kernel_size=3) → 20x24x24
│   └── bn2: BatchNorm2d(20)
├── pool: MaxPool2d(2, 2) → 20x12x12
├── dropout: Dropout2d(0.25)
└── fc1: Linear(2880, 10)
```

### Parameter Breakdown
| Layer | Shape | Parameters |
|-------|-------|------------|
| conv1.weight | [10, 1, 3, 3] | 90 |
| conv1.bias | [10] | 10 |
| bn1.weight | [10] | 10 |
| bn1.bias | [10] | 10 |
| conv2.weight | [20, 10, 3, 3] | 1,800 |
| conv2.bias | [20] | 20 |
| bn2.weight | [20] | 20 |
| bn2.bias | [20] | 20 |
| fc1.weight | [10, 2880] | 28,800 |
| fc1.bias | [10] | 10 |
| **Total** | | **30,790** |

### Training Configuration
- **Optimizer**: Adam (lr=0.01)
- **Scheduler**: StepLR (step_size=15, gamma=0.1)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 128
- **Epochs Trained**: 5

### Results
| Metric | Value | Status |
|--------|-------|--------|
| Total Parameters | 30,790 | ❌ Exceeds by 5,790 |
| Epoch 1 Test Accuracy | 97.73% | ✅ Exceeds goal by 2.73% |
| Best Test Accuracy | 98.61% (Epoch 4) | ✅ |
| Final Test Accuracy | 98.30% (Epoch 5) | ✅ |

### Epoch-wise Performance
| Epoch | Train Acc | Test Acc | Train Loss | Test Loss |
|-------|-----------|----------|------------|-----------|
| 1 | 93.10% | 97.73% | 0.3316 | 0.0006 |
| 2 | 96.57% | 98.01% | 0.1135 | 0.0005 |
| 3 | 96.97% | 98.17% | 0.0987 | 0.0004 |
| 4 | 97.21% | 98.61% | 0.0901 | 0.0004 |
| 5 | 97.33% | 98.30% | 0.0889 | 0.0004 |

### Analysis
**Strengths:**
- ✅ Accuracy target achieved (97.73% in epoch 1 vs 95% goal)
- ✅ Strong generalization (test accuracy exceeds train accuracy)
- ✅ Stable training with consistent improvement

**Issues:**
- ❌ Parameter count: 30,790 (23% over limit)
- ❌ Bottleneck: FC layer (fc1) uses 28,800 params (93.5% of total)

### Problem Identification
The fully connected layer `fc1: Linear(2880, 10)` is consuming **93.5%** of the model's parameters:
- Input features: 20 × 12 × 12 = 2,880
- Output classes: 10
- Parameters: 2,880 × 10 = 28,800

---

## Next Steps for Iteration 2

### Strategy: Reduce FC Layer Parameters

**Root Cause:** The large spatial dimensions (12×12) before flattening create massive FC layer input.

**Solution Options (in priority order):**

1. **Add More Pooling** (Recommended)
   - Add another MaxPool2d(2,2) after conv2
   - This reduces spatial dimensions from 12×12 → 6×6
   - New FC input: 20 × 6 × 6 = 720
   - New FC params: 720 × 10 = 7,200 (saves 21,600 params!)
   - **Expected total params: ~9,190** ✅

2. **Use Global Average Pooling (GAP)**
   - Replace FC layer with AdaptiveAvgPool2d((1,1))
   - Eliminates FC layer entirely
   - Minimal parameter overhead
   - May need to adjust channels in conv2
   - **Expected total params: ~2,000** ✅

3. **Reduce Channels + More Pooling**
   - Change conv2 from 20→16 channels
   - Add MaxPool2d after conv2
   - New FC input: 16 × 6 × 6 = 576
   - New FC params: 576 × 10 = 5,760
   - **Expected total params: ~7,400** ✅

### Recommended Architecture for Iteration 2
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(20 * 6 * 6, 10)  # Changed from 2880 to 720

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(x)  # Add second pooling here
        x = self.dropout(x)
        x = x.view(-1, 20 * 6 * 6)
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)
```

### Expected Impact
- Parameters: ~9,190 (63% reduction, well under 25K limit)
- Accuracy: Should maintain 95%+ (extra pooling acts as regularization)
- Risk: Low (model still has sufficient capacity for MNIST)

### Alternative if Accuracy Drops
If accuracy falls below 95% with double pooling, try:
- Increase conv2 channels: 20 → 24 or 28
- Adjust learning rate or optimizer
- Add one more conv layer before final pooling

---

## Iteration 2

### Architecture Details
```
Model: Net (Double Pooling Architecture)
├── conv1: Conv2d(1, 10, kernel_size=3) → 10x26x26
│   └── bn1: BatchNorm2d(10)
├── conv2: Conv2d(10, 20, kernel_size=3) → 20x24x24
│   └── bn2: BatchNorm2d(20)
├── pool1: MaxPool2d(2, 2) → 20x12x12
├── pool2: MaxPool2d(2, 2) → 20x6x6
├── dropout: Dropout2d(0.25)
└── fc1: Linear(720, 10)
```

### Parameter Breakdown
| Layer | Shape | Parameters |
|-------|-------|------------|
| conv1.weight | [10, 1, 3, 3] | 90 |
| conv1.bias | [10] | 10 |
| bn1.weight | [10] | 10 |
| bn1.bias | [10] | 10 |
| conv2.weight | [20, 10, 3, 3] | 1,800 |
| conv2.bias | [20] | 20 |
| bn2.weight | [20] | 20 |
| bn2.bias | [20] | 20 |
| fc1.weight | [10, 720] | 7,200 |
| fc1.bias | [10] | 10 |
| **Total** | | **9,190** |

### Training Configuration
- **Optimizer**: Adam (lr=0.01)
- **Scheduler**: StepLR (step_size=15, gamma=0.1)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 128
- **Epochs Trained**: 5

### Results
| Metric | Value | Status |
|--------|-------|--------|
| Total Parameters | 9,190 | ✅ Under by 15,810 |
| Epoch 1 Test Accuracy | 97.87% | ✅ Exceeds goal by 2.87% |
| Best Test Accuracy | 98.90% (Epoch 5) | ✅ |
| Final Test Accuracy | 98.90% (Epoch 5) | ✅ |

### Epoch-wise Performance
| Epoch | Train Acc | Test Acc | Train Loss | Test Loss |
|-------|-----------|----------|------------|-----------|
| 1 | 90.91% | 97.87% | 0.3268 | 0.0005 |
| 2 | 96.25% | 98.27% | 0.1229 | 0.0004 |
| 3 | 96.75% | 98.59% | 0.1067 | 0.0004 |
| 4 | 96.95% | 98.15% | 0.0990 | 0.0005 |
| 5 | 97.08% | 98.90% | 0.0933 | 0.0003 |

### Per-Class Accuracy (Epoch 1)
| Class | Accuracy |
|-------|----------|
| 0 | 98.98% |
| 1 | 99.03% |
| 2 | 98.84% |
| 3 | 98.61% |
| 4 | 94.60% |
| 5 | 98.99% |
| 6 | 98.96% |
| 7 | 95.53% |
| 8 | 97.02% |
| 9 | 98.12% |

### Per-Class Accuracy (Epoch 5 - Best)
| Class | Accuracy |
|-------|----------|
| 0 | 99.49% |
| 1 | 99.74% |
| 2 | 99.13% |
| 3 | 98.61% |
| 4 | 99.29% |
| 5 | 99.10% |
| 6 | 98.75% |
| 7 | 98.25% |
| 8 | 98.15% |
| 9 | 98.41% |

### Analysis
**✅ ALL GOALS ACHIEVED!**

**Strengths:**
- ✅ **Parameter efficiency**: 9,190 params (70% reduction from Iteration 1)
- ✅ **Accuracy target exceeded**: 97.87% in epoch 1 (vs 95% goal)
- ✅ **Excellent generalization**: Test accuracy consistently higher than train accuracy
- ✅ **Stable training**: Smooth convergence with no overfitting
- ✅ **Best performance**: 98.90% test accuracy (Epoch 5)
- ✅ **Improved over Iteration 1**: Better final accuracy (98.90% vs 98.30%)

**Key Improvements from Iteration 1:**
| Metric | Iteration 1 | Iteration 2 | Change |
|--------|-------------|-------------|--------|
| Parameters | 30,790 | 9,190 | -70.2% ✅ |
| Epoch 1 Test Acc | 97.73% | 97.87% | +0.14% ✅ |
| Best Test Acc | 98.61% | 98.90% | +0.29% ✅ |
| FC Layer Params | 28,800 (93.5%) | 7,200 (78.3%) | -75.0% ✅ |

**Observations:**
- Additional pooling layer successfully reduced parameters without accuracy loss
- Model shows strong regularization (test > train accuracy throughout)
- Classes 4 and 7 show slightly lower accuracy in epoch 1, but improve significantly by epoch 5
- Model converges smoothly with Adam optimizer
- No signs of overfitting across 5 epochs

### Solution Effectiveness
The **double pooling strategy** successfully addressed the FC layer bottleneck:
- Reduced spatial dimensions: 12×12 → 6×6 (75% reduction in spatial size)
- Reduced FC input features: 2,880 → 720 (75% reduction)
- Reduced FC parameters: 28,800 → 7,200 (75% reduction)
- **Result**: 70% total parameter reduction while maintaining high accuracy

---

## Summary & Recommendations

### Iteration Comparison
| Iteration | Parameters | Epoch 1 Test Acc | Best Test Acc | Goals Met |
|-----------|------------|------------------|---------------|-----------|
| 1 | 30,790 ❌ | 97.73% ✅ | 98.61% | 1/2 |
| 2 | 9,190 ✅ | 97.87% ✅ | 98.90% | 2/2 ✅ |

### Final Status
🎉 **PROJECT COMPLETE - ALL GOALS ACHIEVED!**

**Iteration 2 meets all project requirements:**
- ✅ Parameters: 9,190 < 25,000 (63% under limit)
- ✅ Test Accuracy: 97.87% ≥ 95% in epoch 1 (2.87% above goal)

### Key Learnings
1. **FC layer bottleneck**: In simple CNNs, fully connected layers often dominate parameter count
2. **Pooling as parameter reducer**: Additional pooling effectively reduces spatial dimensions and FC parameters
3. **Regularization benefit**: Extra pooling acts as regularization, improving generalization
4. **Trade-off myth**: Sometimes reducing parameters can actually improve performance (better regularization)

### Future Exploration Ideas (Optional)
If you want to push the model further:
1. **Global Average Pooling (GAP)**: Could reduce to ~2,000 params while potentially maintaining accuracy
2. **More convolutional layers**: Add conv3 layer with more pooling for better feature extraction
3. **Depthwise Separable Convolutions**: Could achieve even better parameter efficiency
4. **Data augmentation tuning**: Already strong, but could experiment with different augmentation strategies
5. **Different optimizers**: Try SGD with momentum for potentially better convergence
