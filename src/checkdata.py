import os, imghdr, sys
from collections import defaultdict

DATA_DIR = r"C:\Users\zhoul\Desktop\340w\data"
SPLITS = ["train", "val", "test"] 

def is_image(path):
    kind = imghdr.what(path)
    return kind in {"jpeg", "png", "bmp", "tiff", "gif"}

def count_images_in_dir(root):
    counts = defaultdict(int)
    bad_files = []
    classes = []
    if not os.path.isdir(root):
        return classes, counts, bad_files
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        classes.append(cls)
        for fn in os.listdir(cls_dir):
            fp = os.path.join(cls_dir, fn)
            if os.path.isfile(fp):
                if is_image(fp):
                    counts[cls] += 1
                else:
                    bad_files.append(fp)
    return classes, counts, bad_files

def main():
    all_split_classes = {}
    ok = True

    print(f"Scanning dataset under: {os.path.abspath(DATA_DIR)}\n")
    # 逐个 split 统计
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        print(f"==> [{split}] dir: {split_dir}")
        if not os.path.isdir(split_dir):
            print(f" Missing split directory: {split_dir}")
            ok = False
            continue
        classes, counts, bad = count_images_in_dir(split_dir)
        all_split_classes[split] = classes

        if not classes:
            print("No class subfolders found.")
            ok = False
        else:
            print(f"   Classes ({len(classes)}): {classes}")
            total = sum(counts.values())
            for c in classes:
                print(f"     - {c}: {counts[c]} images")
            print(f"   Total images: {total}")

        if bad:
            ok = False
            print(f"Found {len(bad)} non-image or corrupted files (first 5 shown):")
            for p in bad[:5]:
                print(f"      - {p}")
        print()

    # 检查三套 split 的类别是否一致
    if len(all_split_classes) == len(SPLITS):
        sets = {k: set(v) for k, v in all_split_classes.items()}
        inter = set.intersection(*sets.values()) if sets else set()
        union = set.union(*sets.values()) if sets else set()
        if inter != union:
            ok = False
            print("Class folders are inconsistent across splits:")
            for k, v in sets.items():
                missing = union - v
                extra   = v - inter
                print(f"   - {k}: missing={sorted(missing)} extra={sorted(extra)}")
        else:
            print(" Class folders are consistent across train/val/test.\n")

    if ok:
        print(" Data check PASSED: structure & files look good.")
        sys.exit(0)
    else:
        print("\n Data check FOUND issues. Please fix the warnings above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
