# src/run_sweep.py
import os
import re
import csv
import time
import subprocess
from pathlib import Path

RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4]

# Try to extract AUC from printed line like:
# [ENT_Avg_q0.7] ... AUC(ovr)=0.9803
AUC_RE = re.compile(r"\[([^\]]+)\].*?AUC\(ovr\)=([0-9.]+)")

def run_one(ratio: float) -> dict:
    out_dir = f"results_prune_{ratio}"
    env = os.environ.copy()
    env["OUT_DIR"] = out_dir
    env["PRUNE_RATIO"] = str(ratio)  # 需要你 config.py 用 env 读 PRUNE_RATIO

    t0 = time.time()

    # 实时打印 pipeline 输出（不再“卡住看不到”）
    proc = subprocess.Popen(
        ["python", "src/pipeline.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stdout_lines = []
    for line in proc.stdout:
        print(line, end="")          # ✅ 实时输出到终端
        stdout_lines.append(line)    # ✅ 仍然保存下来用于解析 AUC

    proc.wait()
    dt = time.time() - t0
    stdout = "".join(stdout_lines)
    stderr = ""

    # Extract AUC
    auc_lines = AUC_RE.findall(stdout)
    auc_map = {name: float(val) for name, val in auc_lines}

    target_key = None
    for k in auc_map.keys():
        if "ENT_Avg_q0.7" in k:
            target_key = k
            break
    if target_key is None:
        for k in auc_map.keys():
            if "ENT_Avg" in k:
                target_key = k
                break

    auc_value = None
    if target_key is not None:
        auc_value = auc_map[target_key]
    elif auc_lines:
        auc_value = float(auc_lines[-1][1])

    return {
        "prune_ratio": ratio,
        "out_dir": out_dir,
        "returncode": proc.returncode,
        "time_sec": round(dt, 2),
        "auc_tag": target_key if target_key else "",
        "auc": auc_value if auc_value is not None else "",
        "stderr_tail": "",
    }


def main():
    # ---- OPTIONAL but recommended ----
    # Make config read PRUNE_RATIO env var (one-line tweak):
    # Cfg.prune_ratio = float(os.getenv("PRUNE_RATIO", "0.3"))
    #
    # If you do that, we can set env["PRUNE_RATIO"] each run automatically.
    # ----------------------------------

    results_dir = Path("results_sweep")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "sweep.csv"

    rows = []
    for r in RATIOS:
        print(f"\n=== RUN prune_ratio={r} ===")
        # If you applied the PRUNE_RATIO env tweak in config.py, uncomment:
        # os.environ["PRUNE_RATIO"] = str(r)

        # Better: pass it via env copy
        row = run_one(r)
        rows.append(row)

        print(f"done: out_dir={row['out_dir']} time_sec={row['time_sec']} auc={row['auc']} tag={row['auc_tag']}")
        if row["returncode"] != 0:
            print("WARNING: pipeline returned non-zero. Check stderr_tail in CSV.")

    # Write CSV
    fieldnames = ["prune_ratio", "out_dir", "returncode", "time_sec", "auc_tag", "auc", "stderr_tail"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\nSaved sweep results to: {csv_path}")


if __name__ == "__main__":
    main()
