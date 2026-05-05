"""
Neural Networks Deep Dive - QuickDraw Dataset
=============================================
Topics covered:
- Weight initialization (uniform, constant, normal)
- Gradient and activation visualization
- Model evaluation & confusion matrix analysis
- Activation functions (ReLU vs Sigmoid)
- Feature map visualization (activation maps)
- Pooling layers (max and average pooling)
- Data augmentation techniques

Dataset: QuickDraw (50M drawings across 345 categories)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()


# ─── Model Architectures ──────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    """Fully-connected network used for initialization experiments."""

    def __init__(self, activation_fn: Callable, n_classes: int, input_size: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, 256),
            nn.Linear(256, 128),
            nn.Linear(128, n_classes),
        ])
        self.activation_fn = activation_fn
        self.input_size = input_size
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)
        return x


class SimpleCNNModel(nn.Module):
    """Shallow CNN used for activation-map and pooling experiments."""

    def __init__(self, n_classes: int, pool_type: str = "max"):
        super().__init__()
        pool = nn.MaxPool2d(2, 2) if pool_type == "max" else nn.AvgPool2d(2, 2)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), pool,
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), pool,
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─── Section 1 – Weight Initialization ────────────────────────────────────────

def visualize_weight_distribution(model: nn.Module):
    """Plot per-layer weight distributions."""
    layer_weight_dict = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            layer_weight_dict[name] = param.detach().cpu().numpy().flatten()

    fig, axes = plt.subplots(1, len(layer_weight_dict), figsize=(15, 4))
    for ax, (name, weights) in zip(axes, layer_weight_dict.items()):
        ax.hist(weights, bins=50, color="steelblue", edgecolor="white")
        ax.set_title(name)
        ax.set_xlabel("Weight values")
    plt.suptitle("Weight distributions per layer")
    plt.tight_layout()
    plt.show()


def const_init(model: nn.Module, c: float | None = None):
    """Initialise all weights to constant value *c* (bias → 0)."""
    if c is not None:
        assert c > 0.0
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data.fill_(c)
        elif "bias" in name:
            param.data.fill_(0.0)


def normal_init(model: nn.Module, std: float = 0.01):
    """Initialise weights from N(0, std²) (bias → 0)."""
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.normal_(param, mean=0.0, std=std)
        elif "bias" in name:
            param.data.fill_(0.0)


# ─── Section 1 – Gradient & Activation Visualization ─────────────────────────

class DummyDataset(Dataset):
    def __init__(self, device=torch.device("cpu")):
        self.X = torch.randn(128, 1, 768, device=device)
        self.y = torch.randint(0, 10, (128,), device=device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"features": self.X[idx], "label": self.y[idx]}


def visualize_gradients(model, data_set, device=torch.device("cpu"), color="C0"):
    """Plot per-layer gradient distributions after one forward+backward pass."""
    model.eval()
    model = model.to(device)
    loader = DataLoader(data_set, batch_size=64, shuffle=True)
    batch = next(iter(loader))

    x = batch["features"].to(device).view(batch["features"].size(0), -1)
    y = batch["label"].to(device)

    model.zero_grad()
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()

    grad_dict = {}
    for i, layer in enumerate(model.layers):
        if layer.weight.grad is not None:
            grad_dict[f"Layer {i}"] = layer.weight.grad.cpu().numpy().flatten()

    fig, axes = plt.subplots(1, len(grad_dict), figsize=(15, 4))
    for ax, (name, grads) in zip(axes, grad_dict.items()):
        ax.hist(grads, bins=50, color=color, edgecolor="white")
        ax.set_title(f"Gradients – {name}")
        ax.set_xlabel("Gradient value")
    plt.suptitle("Gradient distributions per layer")
    plt.tight_layout()
    plt.show()


def visualize_activations(model, data_set, device=torch.device("cpu"), color="C0"):
    """Plot per-layer activation distributions."""
    model.eval()
    model = model.to(device)
    loader = DataLoader(data_set, batch_size=64, shuffle=True)
    batch = next(iter(loader))

    x = batch["features"].to(device).view(batch["features"].size(0), -1)

    activations = {}
    with torch.no_grad():
        out = x
        for i, layer in enumerate(model.layers):
            out = layer(out)
            out = model.activation_fn(out)
            activations[f"Layer {i}"] = out.cpu().numpy().flatten()

    fig, axes = plt.subplots(1, len(activations), figsize=(15, 4))
    for ax, (name, acts) in zip(axes, activations.items()):
        ax.hist(acts, bins=50, color=color, edgecolor="white")
        ax.set_title(f"Activations – {name}")
        ax.set_xlabel("Activation value")
    plt.suptitle("Activation distributions per layer")
    plt.tight_layout()
    plt.show()


# ─── Section 3 – Evaluation & Confusion Matrix ────────────────────────────────

def extract_predictions(model, loader, device):
    """Return (y_true, y_pred) numpy arrays over a DataLoader."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
    return np.array(all_true), np.array(all_pred)


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Plot a normalised confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ─── Section 4 – Activation Functions ─────────────────────────────────────────

def train_and_compare_activations(train_loader, val_loader, n_classes, input_size,
                                   epochs=10, device=torch.device("cpu")):
    """
    Train two SimpleMLP models – one with ReLU, one with Sigmoid –
    and return their training histories for comparison.
    """
    results = {}
    for name, act_fn in [("ReLU", nn.ReLU()), ("Sigmoid", nn.Sigmoid())]:
        model = SimpleMLP(act_fn, n_classes=n_classes, input_size=input_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "val_acc": []}
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                x = batch["features"].to(device)
                y = batch["label"].to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            history["train_loss"].append(epoch_loss / len(train_loader))

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["features"].to(device)
                    y = batch["label"].to(device)
                    preds = model(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += len(y)
            history["val_acc"].append(correct / total)

        results[name] = (model, history)
    return results


def plot_learning_curves(results: dict):
    """Plot training loss and validation accuracy for each model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for name, (_, history) in results.items():
        ax1.plot(history["train_loss"], label=name)
        ax2.plot(history["val_acc"], label=name)
    ax1.set_title("Training loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.set_title("Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.show()


# ─── Section 5 & 6 – Activation Maps & Pooling ────────────────────────────────

def get_activation_maps(model: SimpleCNNModel, x: torch.Tensor):
    """Extract feature maps after the first conv block."""
    maps = {}
    with torch.no_grad():
        out = x
        for i, layer in enumerate(model.features):
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                maps[f"conv_block_{i}"] = out.cpu()
    return maps


def visualize_activation_maps(maps: dict, n_filters: int = 8):
    """Display the first *n_filters* feature maps for each block."""
    for block_name, feature_map in maps.items():
        fig, axes = plt.subplots(1, n_filters, figsize=(2 * n_filters, 3))
        for i, ax in enumerate(axes):
            if i < feature_map.shape[1]:
                ax.imshow(feature_map[0, i].numpy(), cmap="viridis")
                ax.axis("off")
                ax.set_title(f"F{i}")
        fig.suptitle(f"Activation maps – {block_name}")
        plt.tight_layout()
        plt.show()


# ─── Section 7 – Data Augmentation ────────────────────────────────────────────

def get_augmented_transforms():
    """Return a torchvision transform pipeline with typical augmentations."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# ─── Utility ──────────────────────────────────────────────────────────────────

def plot_pixel_intensity_distribution(loader, title="Pixel intensity distribution"):
    """Histogram of raw pixel values across the dataset."""
    all_pixels = []
    for batch in loader:
        all_pixels.append(batch["features"].numpy().flatten())
    all_pixels = np.concatenate(all_pixels)

    plt.figure(figsize=(8, 4))
    plt.hist(all_pixels, bins=100, color="steelblue", edgecolor="none")
    plt.title(title)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_mean_intensity_per_class(loader, class_names, title="Mean intensity per class"):
    """Bar chart of mean pixel intensity for each class."""
    sums = {c: 0.0 for c in class_names}
    counts = {c: 0 for c in class_names}

    for batch in loader:
        feats = batch["features"]
        labels = batch["label"]
        for f, l in zip(feats, labels):
            cname = class_names[l.item()]
            sums[cname] += f.mean().item()
            counts[cname] += 1

    means = [sums[c] / counts[c] if counts[c] else 0 for c in class_names]

    plt.figure(figsize=(10, 4))
    plt.bar(class_names, means, color="steelblue")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Mean pixel intensity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ─── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_CLASSES = 10
    INPUT_SIZE = 768

    # Build models
    simple_mlp = SimpleMLP(nn.ReLU(), n_classes=N_CLASSES, input_size=INPUT_SIZE).to(device)
    const_init_model = SimpleMLP(nn.ReLU(), n_classes=N_CLASSES, input_size=INPUT_SIZE).to(device)
    normal_init_model = SimpleMLP(nn.ReLU(), n_classes=N_CLASSES, input_size=INPUT_SIZE).to(device)

    const_init(const_init_model, c=1.0)
    normal_init(normal_init_model, std=0.01)

    print("Default (uniform) init:")
    visualize_weight_distribution(simple_mlp)

    print("Constant init (c=1):")
    visualize_weight_distribution(const_init_model)

    print("Normal init (std=0.01):")
    visualize_weight_distribution(normal_init_model)

    dummy_ds = DummyDataset(device=device)
    visualize_gradients(simple_mlp, dummy_ds, device=device)
    visualize_activations(simple_mlp, dummy_ds, device=device)