# Influence-Based Pruning and True Network with Entropy-Guided Selective Inference

This repository implements a **data-efficient two-stage inference framework** for ultrasound image classification.  
The proposed system integrates **influence-based dataset pruning (GradNorm proxy)** with a **True Network architecture**, and further introduces an **entropy-based gating mechanism** that selectively refines only uncertain predictions.

The goal is to improve robustness and efficiency by:
- Removing low-impact training samples before training
- Training a secondary *True Network* on high-confidence consensus data
- Invoking the True Network **only when prediction uncertainty is high**

---


---

### Project Overview

Medical ultrasound imaging often presents challenges such as:
- Small and imbalanced datasets  
- Redundant or low-influence training samples  
- High uncertainty in borderline cases  
- High computational cost during retraining  

To address these limitations, this project integrates:

### ✔ Influence-Based Pruning  
Removes 30–40% of low-impact samples while preserving generalization, significantly reducing training time and power usage.

### ✔ True Network (Dual-Model Architecture)  
A secondary classifier re-evaluates uncertain samples produced by the base model.

### ✔ Entropy-Based Gating  
Two additional strategies were introduced:
- **ENT-Replace** — Replace low-confidence logits with outputs from the secondary model  
- **ENT-Avg** — Average logits for more stable uncertainty handling  

This produces a robust, uncertainty-aware ultrasound classification system.

---




> **Important:**  
> The dataset must follow the **ImageFolder format** used by torchvision.

---

## 2. Environment Setup

### 2.1 Create a Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```2.2 Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```