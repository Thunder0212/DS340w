# Influence-Based Pruning and True Network for Efficient Ultrasound Classification

This repository contains the implementation for my DS 340W project, which combines **influence-based data pruning** with an enhanced **True Network** architecture to create a data-efficient and energy-efficient ultrasound classification pipeline.

The system is tested on breast ultrasound datasets and designed to maintain diagnostic performance while reducing redundant training data and computational cost.

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



## Installation & Environment Setup

This project supports **Python 3.11–3.12**.



### Clone the repository

```bash
git clone https://github.com/<your-username>/DS340w.git
```

### Create and activate a virtual environment
```Windows PowerShell
bash

py -3.12 -m venv .venv
.\.venv\Scripts\activate
```
### Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

If GPU installation fails (e.g., RTX 5060 Ti sm_120 is too new), install CPU version: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

### Running the Project

Run the entire pipeline with:


python src/pipeline.py



