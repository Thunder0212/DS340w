import time
from typing import List

import numpy as np
import torch
import torch.nn as nn


def _get_head_params(model: nn.Module) -> List[torch.nn.Parameter]:
    """
    Try to locate classification head parameters for common backbones:
    - torchvision ResNet: model.fc
    - timm ViT: model.head
    - generic: last Linear layers
    """
    if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Module):
        return list(model.fc.parameters())
    if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Module):
        return list(model.head.parameters())

    linear_modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_modules:
        return list(model.parameters())
    return list(linear_modules[-1].parameters())


def compute_gradnorm_scores(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    head_only: bool = True,
    max_samples: int = 0,
    use_amp: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Gradient-Norm Influence Proxy:
    For each example i, compute ||∇_head L_i||_2.
    Returns scores aligned with loader order.

    Notes:
    - loader should be deterministic order (shuffle=False)
    - ideally batch_size=1 for per-sample scores
    """
    model = model.to(device)
    model.train()

    # Select which params to score
    scored_params = _get_head_params(model) if head_only else list(model.parameters())

    # IMPORTANT FIX:
    # Use object identity (id) for membership checking.
    # This avoids tensor equality comparisons that can crash.
    scored_ids = {id(p) for p in scored_params}
    for p in model.parameters():
        p.requires_grad_(id(p) in scored_ids)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    scores: List[float] = []
    t0 = time.time()
    seen = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Expected loader batch to be (x, y)")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Compute L2 norm over chosen params
        total_sq = 0.0
        for p in scored_params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            total_sq += float(torch.sum(g * g).item())

        scores.append(float(np.sqrt(total_sq)))

        seen += 1
        if max_samples and seen >= max_samples:
            break

        if verbose and (seen % 200 == 0):
            print(f"[prune] scored {seen} samples, elapsed {time.time() - t0:.1f}s")

    if verbose:
        print(f"[prune] done scoring {seen} samples, elapsed {time.time() - t0:.1f}s")

    return np.asarray(scores, dtype=np.float32)


def select_keep_indices(
    scores: np.ndarray,
    *,
    prune_ratio: float,
    stable: bool = True,
) -> np.ndarray:
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError("prune_ratio must be in [0, 1).")

    n = int(scores.shape[0])
    keep_n = max(1, int(round(n * (1.0 - prune_ratio))))

    kind = "mergesort" if stable else "quicksort"
    order = np.argsort(scores, kind=kind)  # ascending
    keep = order[-keep_n:]                 # top scores
    keep = np.sort(keep)
    return keep.astype(np.int64)


def save_pruning_artifacts(
    out_dir: str,
    keep_indices: np.ndarray,
    scores: np.ndarray,
) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "prune_keep_indices.npy"), keep_indices)
    np.save(os.path.join(out_dir, "prune_scores.npy"), scores)
    print(f"[prune] saved keep_indices & scores to: {out_dir}")
