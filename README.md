# Influence-Based Pruning and True Network with Entropy-Guided Selective Inference

This repository implements a **data-efficient two-stage inference framework** for medical ultrasound and pathology image classification.
The system integrates **influence-based dataset pruning (GradNorm proxy)** with a **True Network architecture**,
and introduces an **entropy-based gating mechanism** that selectively refines only uncertain predictions.

---

## 1. Project Structure

```text
DS340w/
├─ src/
│  ├─ pipeline.py           # Main training & inference pipeline
│  ├─ plot_results.py       # Generate figures from saved predictions
│  ├─ pruning.py            # Influence-based pruning (GradNorm)
│  ├─ models.py             # Model factory (ResNet / ViT)
│  ├─ utils.py              # Data loading, metrics, transforms
│  └─ config.py             # All experiment configurations
├─ data/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ results/
├─ requirements.txt
└─ README.md
```

---

## 2. Environment Setup

### 2.1 Create Virtual Environment (Windows)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2.2 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Dataset Format

The dataset follows the `ImageFolder` format used by `torchvision`, with **train / val / test** splits and **four classes**:

```text
data/
├─ train/
│  ├─ adenocarcinoma/
│  ├─ large.cell.carcinoma/
│  ├─ normal/
│  └─ squamous.cell.carcinoma/
├─ val/
│  ├─ adenocarcinoma/
│  ├─ large.cell.carcinoma/
│  ├─ normal/
│  └─ squamous.cell.carcinoma/
└─ test/
   ├─ adenocarcinoma/
   ├─ large.cell.carcinoma/
   ├─ normal/
   └─ squamous.cell.carcinoma/
```

Folder names are treated as class labels.  
The validation set is optional and can be ignored if not used by the pipeline.

---

## 4. Run Pipeline

From the project root directory:

```bash
python src/pipeline.py
```

This will automatically:
- Perform influence-based pruning
- Train original models and the True Network
- Evaluate margin-based and entropy-based two-stage inference
- Save predictions and metrics

---

## 5. Generate Figures

After the pipeline finishes:

```bash
python src/plot_results.py
```

Figures will be saved to:

```text
results/figs/
├─ Bar_ACC_F1_AUC.png
├─ ConfMat_ENTReplace.png
└─ ROC_Original_vs_ENTReplace.png
```

---

## 6. Configuration

All experimental settings are defined in `src/config.py`.

Key parameters:

```python
backbone_list = ["resnet18", "resnet34", "resnet50"]
original_backbone = "resnet50"
true_backbone = "resnet50"

prune_ratio = 0.3
use_entropy_gate = True
entropy_quantile = 0.9
threshold_T = 0.15
```

---

## 7. Outputs

- `results/preds/` : saved predictions for plotting
- `results/figs/`  : generated figures
- `results/prune_*.npy` : pruning artifacts

---

## 8. Notes

- GradNorm-based pruning is slower than standard training due to per-sample backpropagation.
- AMP-related warnings can be safely ignored.
- If you change backbone names or dataset structure, delete `results/` before re-running.

---


