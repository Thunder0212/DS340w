import os

class Cfg:
    # 路径
    data_dir =r"C:\Users\zhoul\Desktop\340w\data"
    out_dir  = os.getenv("OUT_DIR",  "results")

    # 模型与训练
    backbone_list = ["resnet50", "resnet101", "vit_base_patch16_224"]  # 用于构建 True-3 的多骨干
    original_backbone = "resnet101"
    true_backbone     = "resnet101"

    img_size   = 224
    batch_size = 48
    epochs     = 10                  
    num_workers = 4
    seed        = 42
    lr = 1e-4

    # 多分类两步法
    # Two-step inference: margin vs entropy gate
    threshold_T      = 0.25     # existing margin threshold
    use_avg_last3    = True

    # Week 11
    use_entropy_gate = True
    entropy_quantile = 0.7      
