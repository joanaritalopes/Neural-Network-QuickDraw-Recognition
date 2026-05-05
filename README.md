# Neural Networks Deep Dive – QuickDraw Dataset

## Overview

This project explores core deep learning concepts using the [QuickDraw dataset](https://quickdraw.withgoogle.com/data) – a collection of 50 million hand drawings across 345 categories. The work spans seven interconnected topics, building from weight initialisation all the way to data augmentation.

---

## Topics Covered

| Section | Topic | Key Techniques |
|---------|-------|---------------|
| 1 | **Weight Initialisation** | Uniform, constant, normal init; gradient & activation distributions |
| 2 | **Model Review** | MLP architecture, training & evaluation pipelines |
| 3 | **Evaluation & Interpretation** | Confusion matrix analysis, class-level accuracy, pixel intensity distributions |
| 4 | **Activation Functions** | ReLU vs Sigmoid – learning dynamics, vanishing gradients |
| 5 | **Activation Maps** | Feature map extraction and visualisation from CNN layers |
| 6 | **Pooling Layers** | Max pooling vs average pooling – effect on feature representations |
| 7 | **Data Augmentation** | Horizontal flip, rotation, affine transforms – impact on generalisation |

---

## Key Findings

- **Weight initialisation matters enormously.** Constant initialisation (all weights = 1) causes symmetry breaking failure – all neurons learn identical features. Normal initialisation with a small std (0.01) can lead to vanishing gradients in deeper networks. PyTorch's default uniform (Kaiming) init is well-calibrated for ReLU networks.

- **ReLU outperforms Sigmoid** in this setting. Sigmoid saturates at extreme values, causing vanishing gradients in early layers and significantly slower convergence.

- **Activation maps** reveal that early convolutional layers respond to low-level features (edges, textures), while deeper layers capture higher-level semantic patterns.

- **Max pooling** preserves the most prominent features and is generally preferred for classification tasks; average pooling produces smoother representations that can reduce noise but may lose fine-grained detail.

- **Data augmentation** reduces overfitting noticeably on the QuickDraw subset, improving validation accuracy by several percentage points without additional data.

---

## Project Structure

```
neural_networks_quickdraw.py   # Main implementation
```

The file is self-contained and covers:
- `SimpleMLP` – fully-connected model for initialisation experiments
- `SimpleCNNModel` – shallow CNN for activation map / pooling experiments
- Weight init functions: `const_init`, `normal_init`
- Visualisation helpers: `visualize_weight_distribution`, `visualize_gradients`, `visualize_activations`, `visualize_activation_maps`
- `extract_predictions`, `plot_confusion_matrix`
- `train_and_compare_activations`, `plot_learning_curves`
- Data augmentation transforms via `get_augmented_transforms`

---

## Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

---

## Usage

```python
from neural_networks_quickdraw import (
    SimpleMLP, get_device,
    const_init, normal_init,
    visualize_weight_distribution,
    visualize_gradients, visualize_activations,
)
import torch.nn as nn

device = get_device()
model = SimpleMLP(nn.ReLU(), n_classes=10, input_size=768).to(device)

# Inspect default (uniform) initialisation
visualize_weight_distribution(model)

# Compare constant vs normal init
from neural_networks_quickdraw import DummyDataset
normal_init(model, std=0.01)
visualize_gradients(model, DummyDataset(device=device), device=device)
```
