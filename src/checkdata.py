#goodbye imghdr R.I.P

import os
from collections import defaultdict
from PIL import Image

DATA_DIR = r"C:\Users\zhoul\Desktop\340w\data"
SPLITS = ["train", "val", "test"]


def is_image(path):
    """
    Check whether a file is a valid image.
    Replaces imghdr (removed in Python 3.13).
    """
    try:
        img = Image.open(path)
        img.verify()   # Verify image integrity
        return True
    except:
        return False


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

        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)

            # check if valid image
            if is_image(fpath):
                counts[cls] += 1
            else:
                bad_files.append(fpath)

    return classes, counts, bad_files


if __name__ == "__main__":
    print("Checking dataset...")

    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        classes, counts, bad = count_images_in_dir(split_dir)

        print(f"\n=== {split.upper()} ===")
        print("Classes:", classes)
        for c in classes:
            print(f"  {c}: {counts[c]} images")

        if bad:
            print("\nInvalid image files detected:")
            for b in bad:
                print("  ", b)
        else:
            print("No invalid files found.")
