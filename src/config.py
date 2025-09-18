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
    epochs     = 10                   # 与论文一致：10 epoch；最后3轮概率取均值
    lr         = 1e-4
    num_workers = 4
    seed        = 42

    # 多分类两步法：用 Top1-Top2 概率差做“置信度”
    # 二分类用 0.9 太高；多分类推荐 0.2~0.3，这里默认 0.25
    threshold_T    = 0.25
    use_avg_last3  = True

    save_probs_each_epoch = False
