# src/quick_demo.py
import os

# 1) 可选：指定数据与输出目录（也可以改成写死路径）
os.environ.setdefault("DATA_DIR", "data")
os.environ.setdefault("OUT_DIR", "results_quick")

# 2) 仅跑 5 个 epoch 的 ResNet50
from config import Cfg
Cfg.epochs = 5
Cfg.backbone_list = ["resnet50"]   # 仅用于显示/一致性，无实际调用
Cfg.original_backbone = "resnet50"

# 3) 复用 pipeline 里的训练函数，但只跑 Original 一次
from pipeline import train_and_dump_probs

if __name__ == "__main__":
    # 只训练并评估一个 Original: ResNet50
    train_and_dump_probs("resnet50", tag="original")

    print("\n================== QUICK DEMO DONE ==================")
    print("已完成：ResNet50 × 5 epoch 训练与测试指标输出")
    print("概率与标签已保存 Labels Saved 。")
