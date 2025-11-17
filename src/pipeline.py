
import os, numpy as np, torch, torch.nn as nn
from tqdm import tqdm
from config import Cfg
from utils import set_seed, get_loaders, ensure_dir, print_metrics
from models import make_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  

# AMP 
if DEVICE == "cuda":
    autocast = torch.amp.autocast("cuda")
    def make_scaler(): return torch.amp.GradScaler("cuda")
else:
    autocast = torch.amp.autocast("cpu")
    def make_scaler(): return torch.amp.GradScaler("cpu")


def _eval_loader_probs(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xs, ys in tqdm(loader, desc="eval", leave=False):
            xs = xs.to(DEVICE, non_blocking=True)
            p = torch.softmax(model(xs), dim=1).cpu().numpy()[0]
            probs.append(p); labels.append(int(ys.item()))
    return np.stack(probs), np.array(labels)


def train_and_dump_probs(backbone: str, tag: str):
    set_seed(Cfg.seed)
    ensure_dir(Cfg.out_dir)
    train_loader, train_eval_loader, val_loader, test_loader, classes = get_loaders(
        Cfg.data_dir, Cfg.img_size, Cfg.batch_size, Cfg.num_workers
    )
    n_classes = len(classes)

    model = make_model(backbone, num_classes=n_classes).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=Cfg.lr)
    crit  = nn.CrossEntropyLoss()
    scaler = make_scaler()

    last3_train_probs, last3_test_probs = [], []
    train_labels_ref, test_labels_ref   = None, None

    for epoch in range(Cfg.epochs):
        model.train()
        for xs, ys in tqdm(train_loader, desc=f"{tag}:{backbone}:train[{epoch+1}/{Cfg.epochs}]"):
            xs = xs.to(DEVICE, non_blocking=True); ys = ys.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast:
                logits = model(xs)
                loss = crit(logits, ys)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

       
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
    test_probs_mean  = np.mean(np.stack(last3_test_probs),  axis=0) if Cfg.use_avg_last3 else last3_test_probs[-1]

    np.save(os.path.join(Cfg.out_dir, f"{tag}_{backbone}_train_probs.npy"),  train_probs_mean)
    np.save(os.path.join(Cfg.out_dir, f"{tag}_{backbone}_train_labels.npy"), train_labels_ref)
    np.save(os.path.join(Cfg.out_dir, f"{tag}_{backbone}_test_probs.npy"),   test_probs_mean)
    np.save(os.path.join(Cfg.out_dir, f"{tag}_{backbone}_test_labels.npy"),  test_labels_ref)

    print_metrics(f"{tag}-{backbone}-TEST", test_labels_ref, test_probs_mean)

# True-3
def build_true3_mask_from_train():
    labels = None
    votes_correct = []
    for b in Cfg.backbone_list[:3]:
        p = np.load(os.path.join(Cfg.out_dir, f"original_{b}_train_probs.npy"))
        y = np.load(os.path.join(Cfg.out_dir, f"original_{b}_train_labels.npy"))
        if labels is None: labels = y
        pred = np.argmax(p, axis=1)
        votes_correct.append((pred == y).astype(int))
    votes_sum = np.stack(votes_correct, axis=0).sum(axis=0)
    mask = votes_sum >= 2
    np.save(os.path.join(Cfg.out_dir, "true3_mask_train.npy"), mask)
    print(f"[True-3] selected {mask.sum()} / {len(mask)} training samples.")
    return mask

#训练 True
def train_true_network(backbone: str, true3_mask):
    set_seed(Cfg.seed); ensure_dir(Cfg.out_dir)
    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset
    from utils import make_transforms

    train_tf, eval_tf = make_transforms(Cfg.img_size)
    full_train_ds = datasets.ImageFolder(os.path.join(Cfg.data_dir, "train"), transform=train_tf)
    n_classes = len(full_train_ds.classes)
    idx = np.where(true3_mask.astype(bool))[0].tolist()
    true_train_ds = Subset(full_train_ds, idx)

    test_ds  = datasets.ImageFolder(os.path.join(Cfg.data_dir, "test"), transform=eval_tf)

    true_train_loader = DataLoader(
        true_train_ds, batch_size=Cfg.batch_size, shuffle=True,
        num_workers=Cfg.num_workers, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=Cfg.num_workers, pin_memory=True, persistent_workers=True
    )

    model = make_model(backbone, num_classes=n_classes).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=Cfg.lr)
    crit  = nn.CrossEntropyLoss()
    scaler = make_scaler()

    last3_test_probs, test_labels_ref = [], None
    for epoch in range(Cfg.epochs):
        model.train()
        for xs, ys in tqdm(true_train_loader, desc=f"TRUE:{backbone}:train[{epoch+1}/{Cfg.epochs}]"):
            xs = xs.to(DEVICE, non_blocking=True); ys = ys.to(DEVICE, non_blocking=True)
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
    np.save(os.path.join(Cfg.out_dir, f"true_{backbone}_test_probs.npy"),  test_probs_mean)
    np.save(os.path.join(Cfg.out_dir, f"true_{backbone}_test_labels.npy"), test_labels_ref)
    print_metrics(f"TRUE-{backbone}-TEST", test_labels_ref, test_probs_mean)

# 两步法推理（多分类：Top1-Top2 概率差阈值）
def two_step_infer(model_original: str, model_true: str, mode="model1"):
    pO = np.load(os.path.join(Cfg.out_dir, f"original_{model_original}_test_probs.npy"))
    y  = np.load(os.path.join(Cfg.out_dir, f"original_{model_original}_test_labels.npy"))
    pT = np.load(os.path.join(Cfg.out_dir, f"true_{model_true}_test_probs.npy"))

    sorted_probs = np.sort(pO, axis=1)[:, ::-1]       # 概率降序
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]  # Top1-Top2 差值（不确定性）
    use_true = margin < Cfg.threshold_T

    out = pO.copy()
    if mode.lower() == "model1":
        out[use_true] = pT[use_true]                  # 直接替换
    else:
        out[use_true] = (pO[use_true] + pT[use_true]) / 2.0  # 平均融合

    print_metrics(f"TwoStep-{mode.upper()}", y, out)

def two_step_infer_entropy(model_original: str, model_true: str,
                           quantile: float = None,
                           mode: str = "replace"):
    """
    Week 11 Novelty:
    Use predictive entropy instead of (Top1-Top2) margin
    to decide which samples go to the True network.
    """
    if quantile is None:
        quantile = getattr(Cfg, "entropy_quantile", 0.7)

    # 1) Load probs and labels
    pO = np.load(os.path.join(Cfg.out_dir, f"original_{model_original}_test_probs.npy"))
    y  = np.load(os.path.join(Cfg.out_dir, f"original_{model_original}_test_labels.npy"))
    pT = np.load(os.path.join(Cfg.out_dir, f"true_{model_true}_test_probs.npy"))

    # 2) Compute predictive entropy for each test sample
    eps = 1e-8
    entropy = -np.sum(pO * np.log(pO + eps), axis=1)   # shape [N]

    # Higher entropy = more uncertain
    thr = np.quantile(entropy, quantile)
    use_true = entropy > thr    # send only the most uncertain samples to True network

    # 3) Combine outputs
    out = pO.copy()
    if mode.lower() == "replace":
        out[use_true] = pT[use_true]
        tag = f"TwoStep-ENT-replace-q{quantile}"
    else:
        out[use_true] = (pO[use_true] + pT[use_true]) / 2.0
        tag = f"TwoStep-ENT-avg-q{quantile}"

    # 4) Evaluate
    print_metrics(tag, y, out)


if __name__ == "__main__":
    # 1) Train multiple Original models (for True-3 voting)
    for b in Cfg.backbone_list[:3]:
        train_and_dump_probs(b, tag="original")

    # 2) Build True-3 mask from training set
    mask_true3 = build_true3_mask_from_train()

    # 3) Train True network on True-3 subset
    train_true_network(Cfg.true_backbone, mask_true3)

    # 4) Two-step inference: baseline margin-based methods
    two_step_infer(Cfg.original_backbone, Cfg.true_backbone, mode="model1")
    two_step_infer(Cfg.original_backbone, Cfg.true_backbone, mode="model2")

    # 5) Week 11 Novelty: entropy-based two-step inference
    if getattr(Cfg, "use_entropy_gate", False):
        two_step_infer_entropy(Cfg.original_backbone, Cfg.true_backbone,
                               quantile=Cfg.entropy_quantile,
                               mode="replace")
        two_step_infer_entropy(Cfg.original_backbone, Cfg.true_backbone,
                               quantile=Cfg.entropy_quantile,
                               mode="avg")
