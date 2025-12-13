import os

class Cfg:
    data_dir = os.getenv("DATA_DIR", "data")
    out_dir  = os.getenv("OUT_DIR",  "results")

    prune_ratio = float(os.getenv("PRUNE_RATIO", "0.3"))
    prune_method = "gradnorm"   # "none" or "gradnorm"
    prune_head_only = True
    prune_max_samples = 800       # 0 = all
    
    backbone_list = ["resnet18", "resnet34", "resnet50"]
    original_backbone = "resnet50"
    true_backbone     = "resnet50"
    num_workers = 0 if os.getenv("OS_CPU_TRAIN", "0") == "1" else 4
    pin_memory = True

    img_size   = 224
    batch_size = 8
    epochs     = 5                 
    num_workers = 4
    seed        = 42
    lr = 1e-4

    
    # Two-step inference
    threshold_T      = 0.15  
    use_avg_last3    = True


    use_entropy_gate = True
    entropy_quantile = 0.9     
