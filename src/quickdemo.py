
import os

#指定数据与输出目录
os.environ.setdefault("DATA_DIR", "data")
os.environ.setdefault("OUT_DIR", "results_quick")


from config import Cfg
Cfg.epochs = 5
Cfg.backbone_list = ["resnet50"]   
Cfg.original_backbone = "resnet50"


from pipeline import train_and_dump_probs

if __name__ == "__main__":
    
    train_and_dump_probs("resnet50", tag="original")

    print("\n================== QUICK DEMO DONE ==================")
    print("已完成：ResNet50 × 5 epoch 训练与测试指标输出")
    print("概率与标签已保存 Labels Saved 。")
