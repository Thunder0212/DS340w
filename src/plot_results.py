import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score


def load_pred(pred_dir: str, name: str):
    y_true = np.load(os.path.join(pred_dir, f"{name}_y_true.npy"))
    y_prob = np.load(os.path.join(pred_dir, f"{name}_y_prob.npy"))
    y_pred = np.load(os.path.join(pred_dir, f"{name}_y_pred.npy"))
    return y_true, y_prob, y_pred


def macro_ovr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_classes = y_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    aucs = []
    for c in range(n_classes):
        
        if y_true_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        aucs.append(auc(fpr, tpr))
    return float(np.mean(aucs)) if aucs else float("nan")


def plot_roc_two_methods(y_true_a, y_prob_a, label_a,
                         y_true_b, y_prob_b, label_b,
                         out_path: str):
    
    n_classes = y_prob_a.shape[1]
    ya = label_binarize(y_true_a, classes=list(range(n_classes)))
    yb = label_binarize(y_true_b, classes=list(range(n_classes)))

    # micro-average
    fpr_a, tpr_a, _ = roc_curve(ya.ravel(), y_prob_a.ravel())
    fpr_b, tpr_b, _ = roc_curve(yb.ravel(), y_prob_b.ravel())
    auc_a = auc(fpr_a, tpr_a)
    auc_b = auc(fpr_b, tpr_b)

    plt.figure()
    plt.plot(fpr_a, tpr_a, label=f"{label_a} (micro-AUC={auc_a:.3f})")
    plt.plot(fpr_b, tpr_b, label=f"{label_b} (micro-AUC={auc_b:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (micro-average)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confmat(y_true, y_pred, out_path: str, title: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar_metrics(methods_metrics, out_path: str):
    # methods_metrics
    names = [m[0] for m in methods_metrics]
    accs  = [m[1] for m in methods_metrics]
    f1s   = [m[2] for m in methods_metrics]
    aucs  = [m[3] for m in methods_metrics]

    x = np.arange(len(names))
    w = 0.25

    plt.figure(figsize=(10, 4))
    plt.bar(x - w, accs, width=w, label="ACC")
    plt.bar(x,      f1s,  width=w, label="F1(macro)")
    plt.bar(x + w,  aucs, width=w, label="AUC(ovr-macro)")

    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.title("Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def list_available_methods(pred_dir: str):
    
    files = glob.glob(os.path.join(pred_dir, "*_y_true.npy"))
    methods = []
    for f in files:
        base = os.path.basename(f)
        name = base.replace("_y_true.npy", "")
        methods.append(name)
    return sorted(set(methods))


if __name__ == "__main__":
    OUT_DIR = os.environ.get("OUT_DIR", "results")
    pred_dir = os.path.join(OUT_DIR, "preds")
    fig_dir  = os.path.join(OUT_DIR, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"找不到 preds 目录：{pred_dir}（先运行 pipeline.py 生成 *_y_true.npy 等文件）")

    methods = list_available_methods(pred_dir)
    print("[found methods]")
    for m in methods:
        print(" -", m)

    #original_resnet50 vs ENT_Replace_q0.9
    main_a = "original_resnet50"
    main_b = "ENT_Replace_q0.9"

    if main_a not in methods:
        raise FileNotFoundError(f"缺少 {main_a} 的预测文件（在 {pred_dir} 里找不到）")
    if main_b not in methods:
        raise FileNotFoundError(f"缺少 {main_b} 的预测文件（在 {pred_dir} 里找不到）")

    
    y_true_a, y_prob_a, y_pred_a = load_pred(pred_dir, main_a)
    y_true_b, y_prob_b, y_pred_b = load_pred(pred_dir, main_b)

    
    roc_path = os.path.join(fig_dir, "ROC_Original_vs_ENTReplace.png")
    plot_roc_two_methods(
        y_true_a, y_prob_a, "Original-ResNet50",
        y_true_b, y_prob_b, "ENT-Replace-q0.9",
        roc_path
    )
    print("[ok] saved:", roc_path)

    #Confusion Matrix
    cm_path = os.path.join(fig_dir, "ConfMat_ENTReplace.png")
    plot_confmat(y_true_b, y_pred_b, cm_path, "Confusion Matrix: ENT-Replace-q0.9")
    print("[ok] saved:", cm_path)

    
    compare_list = [
        ("Original-R50", "original_resnet50"),
        ("TwoStep", "TwoStep_MODEL1"),
        ("ENT-Replace", "ENT_Replace_q0.9"),
        ("ENT-Avg", "ENT_Avg_q0.9"),
    ]

    metrics = []
    for pretty, key in compare_list:
        if key not in methods:
            print(f"[skip] missing: {key}")
            continue
        yt, yp, ypd = load_pred(pred_dir, key)
        acc = float(accuracy_score(yt, ypd))
        f1  = float(f1_score(yt, ypd, average="macro"))
        aucv = float(macro_ovr_auc(yt, yp))
        metrics.append((pretty, acc, f1, aucv))

    bar_path = os.path.join(fig_dir, "Bar_ACC_F1_AUC.png")
    plot_bar_metrics(metrics, bar_path)
    print("[ok] saved:", bar_path)

    print("\nAll done, Folders：", fig_dir)
