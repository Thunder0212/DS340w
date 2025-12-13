import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Cfg
from utils import set_seed, get_loaders, ensure_dir, print_metrics
from models import make_model
from pruning import compute_gradnorm_scores, select_keep_indices, save_pruning_artifacts


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# AMP
if DEVICE == "cuda":
    autocast = torch.amp.autocast("cuda")
    def make_scaler(): return torch.amp.GradScaler("cuda")
else:
    autocast = torch.amp.autocast("cpu")
    def make_scaler(): return torch.amp.GradScaler("cpu")



def _save_preds(method_name: str, y_true: np.ndarray, y_prob: np.ndarray):
    """Save predictions for plotting (ROC/ConfMat/etc)."""
    pred_dir = os.path.join(Cfg.out_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)
    y_pred = np.argmax(y_prob, axis=1)

    np.save(os.path.join(pred_dir, f"{method_name}_y_true.npy"), y_true)
    np.save(os.path.join(pred_dir, f"{method_name}_y_prob.npy"), y_prob)
    np.save(os.path.join(pred_dir, f"{method_name}_y_pred.npy"), y_pred)


def _eval_loader_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xs, ys in tqdm(loader, desc="eval", leave=False):
            xs = xs.to(DEVICE, non_blocking=True)
            p = torch.softmax(model(xs), dim=1).cpu().numpy()[0]
            probs.append(p)
            labels.append(int(ys.item()))
    return np.stack(probs), np.array(labels)



def compute_pruning_indices_once(classes):
    """
    Returns:
      keep_full_indices: np.ndarray of indices INTO FULL training set order
    """
    ensure_dir(Cfg.out_dir)

   
    train_loader, train_eval_loader, val_loader, test_loader, _classes = get_loaders(
        Cfg.data_dir, Cfg.img_size, Cfg.batch_size, Cfg.num_workers,
        train_indices=None
    )

    if getattr(Cfg, "prune_method", "none") != "gradnorm" or getattr(Cfg, "prune_ratio", 0.0) <= 0.0:
        full_n = len(train_eval_loader.dataset)
        keep = np.arange(full_n, dtype=np.int64)
        save_pruning_artifacts(Cfg.out_dir, keep, np.ones(full_n, dtype=np.float32))
        print("[prune] disabled (keep all)")
        return keep

    print(f"[prune] ENABLED: method=gradnorm, prune_ratio={Cfg.prune_ratio}")

    # scoring model
    score_backbone = Cfg.backbone_list[0]
    score_model = make_model(score_backbone, num_classes=len(classes)).to(DEVICE)

    scores = compute_gradnorm_scores(
        model=score_model,
        loader=train_eval_loader,              
        device=torch.device(DEVICE),
        head_only=getattr(Cfg, "prune_head_only", True),
        max_samples=getattr(Cfg, "prune_max_samples", 0),
        use_amp=False,
        verbose=True,
    )

    keep = select_keep_indices(scores, prune_ratio=Cfg.prune_ratio)
    save_pruning_artifacts(Cfg.out_dir, keep, scores)

    print(f"[prune] kept {len(keep)}/{len(scores)} samples ({len(keep)/len(scores):.2%})")
    return keep



def train_and_dump_probs(backbone: str, tag: str, train_indices_full=None):
    """
    train_indices_full:
      - list of indices in FULL training set order (ImageFolder order)
      - passed into get_loaders(...) which will use Subset(...)
    """
    set_seed(Cfg.seed)
    ensure_dir(Cfg.out_dir)

    train_loader, train_eval_loader, val_loader, test_loader, classes = get_loaders(
        Cfg.data_dir, Cfg.img_size, Cfg.batch_size, Cfg.num_workers,
        train_indices=None if train_indices_full is None else list(map(int, train_indices_full))
    )
    n_classes = len(classes)

    model = make_model(backbone, num_classes=n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=Cfg.lr)
    crit = nn.CrossEntropyLoss()
    scaler = make_scaler()

    last3_train_probs, last3_test_probs = [], []
    train_labels_ref, test_labels_ref = None, None

    for epoch in range(Cfg.epochs):
        model.train()
        for xs, ys in tqdm(train_loader, desc=f"{tag}:{backbone}:train[{epoch+1}/{Cfg.epochs}]"):
            xs = xs.to(DEVICE, non_blocking=True)
            ys = ys.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast:
                logits = model(xs)
                loss = crit(logits, ys)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # dump probs on last 3 epochs
        if epoch >= Cfg.epochs - 3:
            tr_probs, tr_labels = _eval_loader_probs(model, train_eval_loader)
            te_probs, te_labels = _eval_loader_probs(model, test_loader)
            train_labels_ref, test_labels_ref = tr_labels, te_labels
            last3_train_probs.append(tr_probs)
            last3_test_probs.append(te_probs)

    if not last3_train_probs or not last3_test_probs:
        tr_probs, tr_labels = _eval_loader_probs(model, train_eval_loader)
        te_probs, te_labels = _eval_loader_probs(model, test_loader)
        train_labels_ref, test_labels_ref = tr_labels, te_labels
        last3_train_probs, last3_test_probs = [tr_probs], [te_probs]

    train_probs_mean = np.mean(np.stack(last3_train_probs), axis=0) if Cfg.use_avg_last3 else last3_train_probs[-1]
    test_probs_mean = np.mean(np.stack(last3_test_probs), axis=0) if Cfg.use_avg_last3 else last3_test_probs[-1]

    np.save(os.path.join(Cfg.out_dir, f"{tag}_{backbone}_train_probs.npy"), train_probs_mean)
    np.save(os.path.join(Cfg.out_dir, f"{tag}_{backbone}_train_labels.npy"), train_labels_ref)
    np.save(os.path.join(Cfg.out_dir, f"{tag}_{backbone}_test_probs.npy"), test_probs_mean)
    np.save(os.path.join(Cfg.out_dir, f"{tag}_{backbone}_test_labels.npy"), test_labels_ref)

    print_metrics(f"{tag}-{backbone}-TEST", test_labels_ref, test_probs_mean)
    _save_preds(f"{tag}_{backbone}", test_labels_ref, test_probs_mean)



def build_true3_mask_from_train():
    labels = None
    votes_correct = []

    for b in Cfg.backbone_list[:3]:
        p = np.load(os.path.join(Cfg.out_dir, f"original_{b}_train_probs.npy"))
        y = np.load(os.path.join(Cfg.out_dir, f"original_{b}_train_labels.npy"))
        if labels is None:
            labels = y
        pred = np.argmax(p, axis=1)
        votes_correct.append((pred == y).astype(int))

    votes_sum = np.stack(votes_correct, axis=0).sum(axis=0)

    # True-3
    mask = votes_sum >= 2

    
    if mask.sum() == 0:
        print("[True-3] WARNING: 0 samples selected. Fallback to >=1 correct vote.")
        mask = votes_sum >= 1
        print(f"[True-3 fallback] selected {mask.sum()} / {len(mask)} samples.")
    

    np.save(os.path.join(Cfg.out_dir, "true3_mask_train.npy"), mask)
    print(f"[True-3] final selected {mask.sum()} / {len(mask)} samples (ON PRUNED SUBSET ORDER).")
    return mask



def train_true_network(backbone: str, true3_mask_subset: np.ndarray, keep_full_indices: np.ndarray):
    """
    true3_mask_subset: mask over PRUNED subset order (length = kept samples)
    keep_full_indices: indices into FULL train set order (length = kept samples)
    """
    set_seed(Cfg.seed)
    ensure_dir(Cfg.out_dir)

    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset
    from utils import make_transforms

    train_tf, eval_tf = make_transforms(Cfg.img_size)

    full_train_ds = datasets.ImageFolder(os.path.join(Cfg.data_dir, "train"), transform=train_tf)
    n_classes = len(full_train_ds.classes)

    keep_full_indices = np.asarray(keep_full_indices, dtype=np.int64)
    subset_selected = keep_full_indices[true3_mask_subset.astype(bool)].tolist()

    true_train_ds = Subset(full_train_ds, subset_selected)

    if len(true_train_ds) == 0:
        raise RuntimeError(
            "[TRUE] true_train_ds is empty (0 samples). "
            "Increase epochs / adjust pruning / relax True-3 selection."
        )

    test_ds = datasets.ImageFolder(os.path.join(Cfg.data_dir, "test"), transform=eval_tf)

    true_train_loader = DataLoader(
        true_train_ds,
        batch_size=Cfg.batch_size,
        shuffle=True,
        num_workers=Cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=Cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = make_model(backbone, num_classes=n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=Cfg.lr)
    crit = nn.CrossEntropyLoss()
    scaler = make_scaler()

    last3_test_probs, test_labels_ref = [], None

    for epoch in range(Cfg.epochs):
        model.train()
        for xs, ys in tqdm(true_train_loader, desc=f"TRUE:{backbone}:train[{epoch+1}/{Cfg.epochs}]"):
            xs = xs.to(DEVICE, non_blocking=True)
            ys = ys.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast:
                logits = model(xs)
                loss = crit(logits, ys)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if epoch >= Cfg.epochs - 3:
            te_probs, te_labels = _eval_loader_probs(model, test_loader)
            test_labels_ref = te_labels
            last3_test_probs.append(te_probs)

    if not last3_test_probs:
        te_probs, te_labels = _eval_loader_probs(model, test_loader)
        test_labels_ref = te_labels
        last3_test_probs = [te_probs]

    test_probs_mean = np.mean(np.stack(last3_test_probs), axis=0) if Cfg.use_avg_last3 else last3_test_probs[-1]
    np.save(os.path.join(Cfg.out_dir, f"true_{backbone}_test_probs.npy"), test_probs_mean)
    np.save(os.path.join(Cfg.out_dir, f"true_{backbone}_test_labels.npy"), test_labels_ref)

    print_metrics(f"TRUE-{backbone}-TEST", test_labels_ref, test_probs_mean)
    _save_preds(f"TRUE_{backbone}", test_labels_ref, test_probs_mean)



def two_step_infer(model_original: str, model_true: str, mode="model1"):
    pO = np.load(os.path.join(Cfg.out_dir, f"original_{model_original}_test_probs.npy"))
    y = np.load(os.path.join(Cfg.out_dir, f"original_{model_original}_test_labels.npy"))
    pT = np.load(os.path.join(Cfg.out_dir, f"true_{model_true}_test_probs.npy"))

    sorted_probs = np.sort(pO, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    use_true = margin < Cfg.threshold_T

    out = pO.copy()
    if mode.lower() == "model1":
        out[use_true] = pT[use_true]
        tag = "TwoStep_MODEL1"
    else:
        out[use_true] = (pO[use_true] + pT[use_true]) / 2.0
        tag = "TwoStep_MODEL2"

    print_metrics(tag, y, out)
    _save_preds(tag, y, out)



def two_step_infer_entropy(model_original: str, model_true: str,
                           quantile: float = None,
                           mode: str = "replace"):
    if quantile is None:
        quantile = getattr(Cfg, "entropy_quantile", 0.7)

    pO = np.load(os.path.join(Cfg.out_dir, f"original_{model_original}_test_probs.npy"))
    y = np.load(os.path.join(Cfg.out_dir, f"original_{model_original}_test_labels.npy"))
    pT = np.load(os.path.join(Cfg.out_dir, f"true_{model_true}_test_probs.npy"))

    eps = 1e-8
    entropy = -np.sum(pO * np.log(pO + eps), axis=1)
    thr = np.quantile(entropy, quantile)
    use_true = entropy > thr

    out = pO.copy()
    if mode.lower() == "replace":
        out[use_true] = pT[use_true]
        tag = f"ENT_Replace_q{quantile}"
    else:
        out[use_true] = (pO[use_true] + pT[use_true]) / 2.0
        tag = f"ENT_Avg_q{quantile}"

    print_metrics(tag, y, out)
    _save_preds(tag, y, out)



if __name__ == "__main__":
    ensure_dir(Cfg.out_dir)

    _train_loader, _train_eval_loader, _val_loader, _test_loader, classes = get_loaders(
        Cfg.data_dir, Cfg.img_size, Cfg.batch_size, Cfg.num_workers, train_indices=None
    )

    keep_full_indices = compute_pruning_indices_once(classes)

    for b in Cfg.backbone_list[:3]:
        train_and_dump_probs(b, tag="original", train_indices_full=keep_full_indices)

    mask_true3_subset = build_true3_mask_from_train()

    train_true_network(Cfg.true_backbone, mask_true3_subset, keep_full_indices)

    two_step_infer(Cfg.original_backbone, Cfg.true_backbone, mode="model1")
    two_step_infer(Cfg.original_backbone, Cfg.true_backbone, mode="model2")

    if getattr(Cfg, "use_entropy_gate", False):
        two_step_infer_entropy(Cfg.original_backbone, Cfg.true_backbone,
                               quantile=Cfg.entropy_quantile,
                               mode="replace")
        two_step_infer_entropy(Cfg.original_backbone, Cfg.true_backbone,
                               quantile=Cfg.entropy_quantile,
                               mode="avg")
