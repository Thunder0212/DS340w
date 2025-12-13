# Influence-Based Pruning and True Network with Entropy-Guided Selective Inference

This repository implements a **data-efficient two-stage inference framework** for ultrasound image classification.
The proposed system integrates **influence-based dataset pruning (GradNorm proxy)** with a **True Network architecture**,
and introduces an **entropy-based gating mechanism** that selectively refines only uncertain predictions.

## 1. Project Structure

DS340w/
├─ src/
│  ├─ pipeline.py
│  ├─ plot_results.py
│  ├─ pruning.py
│  ├─ models.py
│  ├─ utils.py
│  └─ config.py
├─ data/
│  ├─ train/
│  └─ test/
├─ results/
├─ requirements.txt
└─ README.md

## 2. Environment Setup

### 2.1 Create Virtual Environment (Windows)

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

### 2.2 Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

If timm is missing:
pip install timm

## 3. Dataset Format

data/
├─ train/
│  ├─ class0/
│  └─ class1/
└─ test/
   ├─ class0/
   └─ class1/

## 4. Run Pipeline

python src/pipeline.py

## 5. Generate Figures

python src/plot_results.py

Figures are saved to results/figs/.

## 6. Configuration

Edit src/config.py to change:
- backbone_list
- prune_ratio
- entropy_quantile
- threshold_T

## 7. Output

- results/preds/: saved predictions
- results/figs/: figures
- results/prune_*.npy: pruning artifacts

## 8. Citation

Luping Zhou, Influence-Based Pruning and True Network with Entropy-Guided Selective Inference, 2025.
