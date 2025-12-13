
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score



def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def make_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    return train_tf, eval_tf



def _make_weighted_sampler(targets: List[int]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to reduce class imbalance.
    targets: list of class indices (len = num_samples)
    """
    targets = np.asarray(targets, dtype=np.int64)
    class_counts = np.bincount(targets)
    class_counts = np.maximum(class_counts, 1)  # avoid div-by-zero
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def get_loaders(
    data_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    *,
    train_indices: Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Returns:
      train_loader: training loader (optionally on Subset)
      train_eval_loader: deterministic loader over SAME training set (batch=1, no shuffle)
      val_loader
      test_loader
      classes: class names in consistent order
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_tf, eval_tf = make_transforms(img_size)

    # Base datasets
    train_base_for_train = datasets.ImageFolder(str(train_dir), transform=train_tf)
    train_base_for_eval  = datasets.ImageFolder(str(train_dir), transform=eval_tf)
    val_ds  = datasets.ImageFolder(str(val_dir), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)

    classes = train_base_for_train.classes

    # Optional pruning subset
    if train_indices is not None:
        # Ensure stable ordering
        train_indices = list(map(int, train_indices))

        train_ds = Subset(train_base_for_train, train_indices)
        train_eval_ds = Subset(train_base_for_eval, train_indices)

        # Recompute targets for sampler on subset
        subset_targets = [train_base_for_train.targets[i] for i in train_indices]
        sampler = _make_weighted_sampler(subset_targets)

    else:
        train_ds = train_base_for_train
        train_eval_ds = train_base_for_eval

        sampler = _make_weighted_sampler(train_base_for_train.targets)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # sampler and shuffle are mutually exclusive; keep shuffle False
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, train_eval_loader, val_loader, test_loader, classes



# Metrics

def _specificity_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """
    Macro-average specificity for multi-class:
    specificity_c = TN / (TN + FP) in one-vs-rest setting.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    specs = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        denom = (tn + fp)
        specs.append(0.0 if denom == 0 else tn / denom)
    return float(np.mean(specs))


def metrics_multiclass(y_true: np.ndarray, y_prob: np.ndarray):
    """
    y_true: (N,)
    y_prob: (N, C) probabilities
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    n_classes = int(y_prob.shape[1])
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    sen = recall_score(y_true, y_pred, average="macro")
    spe = _specificity_macro(y_true, y_pred, n_classes)

    # AUC
    try:
        if n_classes == 2:
            aucv = roc_auc_score(y_true, y_prob[:, 1])
        else:
            aucv = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
                labels=list(range(n_classes)),
            )
    except Exception:
        aucv = float("nan")

    return acc, sen, spe, f1, aucv


def print_metrics(tag: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    acc, sen, spe, f1, aucv = metrics_multiclass(y_true, y_prob)
    print(
        f"[{tag}] ACC={acc:.4f} "
        f"SEN(macro)={sen:.4f} "
        f"SPE(macro)={spe:.4f} "
        f"F1(macro)={f1:.4f} "
        f"AUC(ovr)={aucv:.4f}"
    )
