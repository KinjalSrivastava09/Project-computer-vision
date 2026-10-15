"""Reproducible computer vision experiment on synthetic geometric shapes."""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ('circle', 'square', 'triangle')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_image(kind, rng, noise=0.075, image_size=32):
    scale = 4
    s = image_size * scale
    image = Image.new('L', (s, s), 0)
    draw = ImageDraw.Draw(image)
    cx, cy = rng.integers(12, 20, size=2) * scale
    radius = int(rng.integers(6, 11)) * scale
    intensity = int(rng.integers(175, 256))
    if kind == 0:
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=intensity)
    elif kind == 1:
        draw.rectangle((cx-radius, cy-radius, cx+radius, cy+radius), fill=intensity)
    else:
        draw.polygon([(cx, cy-radius), (cx-radius, cy+radius), (cx+radius, cy+radius)], fill=intensity)
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    a = np.asarray(image, dtype=np.float32) / 255.0
    a = np.clip(a + rng.normal(0, noise, a.shape), 0, 1)
    return a.astype(np.float32)


def make_dataset(per_class=700, seed=42, noise=0.075):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for kind in range(3):
        for _ in range(per_class):
            images.append(make_image(kind, rng, noise))
            labels.append(kind)
    return np.stack(images)[:, None, :, :], np.asarray(labels, dtype=np.int64)


class ShapeCNN(nn.Module):
    def __init__(self, pooling=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2) if pooling else nn.Identity(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2) if pooling else nn.Identity(),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * (8 if pooling else 32)**2, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, x):
        return self.classifier(self.features(x))


def train(model, train_loader, test_loader, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct, seen = 0., 0, 0
        for xx, yy in train_loader:
            xx, yy = xx.to(device), yy.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xx)
            loss = loss_fn(logits, yy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yy)
            correct += (logits.argmax(1) == yy).sum().item()
            seen += len(yy)
        eval_res = evaluate(model, test_loader, device)
        row = {'epoch': epoch, 'train_loss': total_loss / seen, 'train_accuracy': correct / seen,
               'test_accuracy': eval_res['accuracy']}
        history.append(row)
        print(f"Epoch {epoch:02d}/{epochs}: loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} test_acc={row['test_accuracy']:.4f}", flush=True)
    return history, round(time.perf_counter() - start, 3)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    true, pred = [], []
    for xx, yy in loader:
        logits = model(xx.to(device))
        pred.extend(logits.argmax(1).cpu().tolist())
        true.extend(yy.tolist())
    true, pred = np.array(true), np.array(pred)
    return {'accuracy': float(np.mean(true == pred)), 'confusion_matrix': confusion_matrix(true, pred, labels=[0, 1, 2]).tolist(),
            'classification_report': classification_report(true, pred, labels=[0, 1, 2], target_names=CLASSES, output_dict=True, zero_division=0)}


def sobel(image):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = kx.T
    padded = np.pad(image, 1)
    gx, gy = np.zeros_like(image), np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(patch * kx)
            gy[i, j] = np.sum(patch * ky)
    return gx, gy, np.sqrt(gx**2 + gy**2)


def make_plots(out, images, labels, history, cm, model, device, noise_metrics, ablation_metrics):
    fig, ax = plt.subplots(2, 6, figsize=(12, 4.5))
    for row in range(2):
        for j in range(6):
            idx = np.flatnonzero(labels == j % 3)[row * 2]
            ax[row, j].imshow(images[idx, 0], cmap='gray', vmin=0, vmax=1)
            ax[row, j].set_title(CLASSES[j % 3].title(), fontsize=10)
            ax[row, j].axis('off')
    fig.tight_layout(); fig.savefig(out/'dataset_samples.png', dpi=150); plt.close(fig)

    gx, gy, mag = sobel(images[0, 0])
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for axis, item, name in zip(axes, [images[0, 0], gx, gy, mag], ['Input', 'Sobel X', 'Sobel Y', 'Magnitude']):
        axis.imshow(item, cmap='gray'); axis.set_title(name); axis.axis('off')
    fig.tight_layout(); fig.savefig(out/'sobel_pipeline.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x['epoch'] for x in history], [x['train_loss'] for x in history], marker='o', label='Training loss')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training loss'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'training_loss.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x['epoch'] for x in history], [x['train_accuracy'] for x in history], marker='o', label='Train')
    ax.plot([x['epoch'] for x in history], [x['test_accuracy'] for x in history], marker='s', label='Test')
    ax.set(xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1.05), title='Accuracy by epoch'); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'accuracy.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5)); im = ax.imshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(np.asarray(cm)):
        ax.text(j, i, str(val), ha='center', va='center', color='white' if val > max(map(max, cm)) / 2 else 'black')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES, xlabel='Predicted', ylabel='True', title='Held-out test confusion matrix')
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out/'confusion_matrix.png', dpi=150); plt.close(fig)

    model.eval()
    with torch.no_grad():
        first = model.features[:2](torch.from_numpy(images[0:1]).to(device)).cpu().numpy()[0]
    fig, axs = plt.subplots(2, 4, figsize=(9, 5))
    for ax, fmap in zip(axs.flat, first):
        ax.imshow(fmap, cmap='viridis'); ax.axis('off')
    fig.suptitle('Learned first-layer feature maps (one example)'); fig.tight_layout()
    fig.savefig(out/'feature_maps.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([p['noise_std'] for p in noise_metrics], [p['accuracy'] for p in noise_metrics], marker='o')
    ax.set(xlabel='Gaussian noise standard deviation', ylabel='Accuracy', ylim=(0, 1.05), title='Noise stress test (fresh generated images)'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'noise_robustness.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['With pooling', 'Without pooling'], [ablation_metrics['pooled_params'], ablation_metrics['unpooled_params']])
    ax.set(ylabel='Trainable parameters', title='Exact parameter count comparison')
    fig.tight_layout(); fig.savefig(out/'parameter_comparison.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Fully reproducible synthetic-shape CNN experiment')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ablation-epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.epochs < 1 or args.ablation_epochs < 1: parser.error('epochs must be >= 1')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed_all(args.seed)
    out = args.output
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    images, labels = make_dataset(seed=args.seed)
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=.22, random_state=args.seed, stratify=labels)
    train_data = TensorDataset(torch.from_numpy(images[train_idx]), torch.from_numpy(labels[train_idx]))
    test_data = TensorDataset(torch.from_numpy(images[test_idx]), torch.from_numpy(labels[test_idx]))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    test_loader = DataLoader(test_data, batch_size=128)
    device = torch.device('cpu')
    pooled = ShapeCNN(True).to(device)
    unpooled = ShapeCNN(False).to(device)
    params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Primary model', params(pooled), 'parameters', flush=True)
    history, elapsed = train(pooled, train_loader, test_loader, args.epochs, .001, device)
    primary = evaluate(pooled, test_loader, device)
    torch.save({'state_dict': pooled.state_dict(), 'classes': CLASSES, 'seed': args.seed}, out/'model.pt')
    print('No-pooling ablation', params(unpooled), 'parameters', flush=True)
    ablation_history, ablation_time = train(unpooled, train_loader, test_loader, args.ablation_epochs, .001, device)
    ablation = evaluate(unpooled, test_loader, device)
    noise_tests = []
    for sigma in (0., .075, .15, .25, .4):
        xi, yi = make_dataset(per_class=100, seed=args.seed+100, noise=sigma)
        rr = evaluate(pooled, DataLoader(TensorDataset(torch.from_numpy(xi), torch.from_numpy(yi)), batch_size=128), device)
        noise_tests.append({'noise_std': sigma, 'accuracy': rr['accuracy'], 'n': len(yi)})
    comparison = {'pooled_params': params(pooled), 'unpooled_params': params(unpooled), 'pooled_flatten_features': 1024,
                  'unpooled_flatten_features': 16384, 'no_pool_accuracy': ablation['accuracy'], 'no_pool_epochs': args.ablation_epochs,
                  'no_pool_training_seconds': ablation_time}
    make_plots(figures, images, labels, history, primary['confusion_matrix'], pooled, device, noise_tests, comparison)
    results = {'experiment': 'Synthetic three-shape recognition', 'seed': args.seed, 'samples_total': len(labels),
               'train_samples': len(train_idx), 'test_samples': len(test_idx), 'test_counts': np.bincount(labels[test_idx]).tolist(),
               'image_shape': [1, 32, 32], 'classes': CLASSES, 'epochs': args.epochs, 'batch_size': args.batch_size,
               'optimizer': 'Adam', 'learning_rate': .001, 'train_seconds': elapsed, 'model_parameters': params(pooled),
               'history': history, 'test': primary, 'ablation': comparison, 'ablation_history': ablation_history,
               'noise_stress_test': noise_tests, 'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device)}}
    (out/'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('Saved', out/'metrics.json', 'test accuracy:', primary['accuracy'], flush=True)


if __name__ == '__main__':
    main()
