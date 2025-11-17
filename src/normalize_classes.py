import os, shutil
from pathlib import Path


DATA_DIR =r"C:\Users\zhoul\Desktop\340w\data"


CANONICAL = {
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "adenocarcinoma",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "large.cell.carcinoma",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "squamous.cell.carcinoma",
    "adenocarcinoma": "adenocarcinoma",
    "large.cell.carcinoma": "large.cell.carcinoma",
    "squamous.cell.carcinoma": "squamous.cell.carcinoma",
    "normal": "normal",
}

SPLITS = ["train", "val", "test"]

def normalize_split(split):
    src_split = Path(DATA_DIR) / split
    if not src_split.is_dir():
        print(f"Missing split: {src_split}")
        return

   
    dst_split = Path(DATA_DIR) / f"{split}_std"
    dst_split.mkdir(parents=True, exist_ok=True)

    
    for cname in ["adenocarcinoma", "large.cell.carcinoma", "squamous.cell.carcinoma", "normal"]:
        (dst_split / cname).mkdir(parents=True, exist_ok=True)

    
    for cls in sorted(os.listdir(src_split)):
        cls_dir = src_split / cls
        if not cls_dir.is_dir():
            continue

        target = CANONICAL.get(cls, None)
        if target is None:
            print(f"❓ Unknown class folder: {cls} (skip)")
            continue

        dst_cls_dir = dst_split / target
        # 拷贝图片
        count = 0
        for fname in os.listdir(cls_dir):
            src_file = cls_dir / fname
            if src_file.is_file():
                shutil.copy2(src_file, dst_cls_dir / fname)
                count += 1
        print(f"[{split}] {cls}  ->  {target}   ({count} files)")

    print(f"✅ Done: {split} → {dst_split}")

def main():
    print(f"Normalizing dataset under: {Path(DATA_DIR).resolve()}\n")
    for sp in SPLITS:
        normalize_split(sp)

    # 提示如何切换到标准化目录
    print("\n下一步：请把标准化目录重命名/接管为正式目录：")
    print("  - 把 train_std 重命名为 train")
    print("  - 把 val_std   重命名为 val")
    print("  - 把 test_std  重命名为 test")
    print("\n或在运行脚本/配置里把数据路径指向 *_std 目录。")

if __name__ == "__main__":
    main()
