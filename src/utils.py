import os, random, numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    return train_tf, eval_tf

def _make_train_sampler(train_ds):
    # 类不平衡过采样（每类样本数的倒数作为权重）
    ys = [y for _, y in train_ds.samples]
    counts = np.bincount(ys)
    class_w = 1.0 / np.maximum(counts, 1)
    weights = [class_w[y] for y in ys]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def get_loaders(data_dir, img_size, batch_size, num_workers):
    train_tf, eval_tf = make_transforms(img_size)
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_tf)

    sampler = _make_train_sampler(train_ds)
    train_loader      = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    train_eval_loader = DataLoader(
        datasets.ImageFolder(os.path.join(data_dir, "train"), transform=eval_tf),
        batch_size=1, shuffle=False, num_workers=num_workers
    )
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1,         shuffle=False, num_workers=num_workers)
    classes = train_ds.classes  # 例如：['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
    return train_loader, train_eval_loader, val_loader, test_loader, classes

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# —— 多分类指标 —— #
def _specificity_macro(y_true, y_pred, n_classes):
    # 针对每一类当作“正类”计算二分类混淆，再做macro平均
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))  # [C,C]
    specs = []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp + 1e-12)
        specs.append(spec)
    return float(np.mean(specs))

def metrics_multiclass(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)         # [N, C]
    n_classes = y_prob.shape[1]
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro')
    sen = recall_score(y_true, y_pred, average='macro')      # 宏平均召回=宏平均Sensitivity
    spe = _specificity_macro(y_true, y_pred, n_classes)

    # AUC（多分类）：一对多 ovr
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=list(range(n_classes)))
    except Exception:
        auc = float("nan")
    return acc, sen, spe, f1, auc

def print_metrics(tag, y_true, y_prob):
    acc, sen, spe, f1, auc = metrics_multiclass(y_true, y_prob)
    print(f"[{tag}] ACC={acc:.4f} SEN(macro)={sen:.4f} SPE(macro)={spe:.4f} F1(macro)={f1:.4f} AUC(ovr)={auc:.4f}")
