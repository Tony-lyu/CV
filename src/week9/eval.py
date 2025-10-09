import argparse, glob, os
import pandas as pd

def load_last_epoch(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    df = df.sort_values("epoch")
    last = df.iloc[-1].copy()
    last["source_file"] = os.path.basename(csv_path)
    return last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default="reports/week9_*.csv", help="Glob for input CSVs")
    ap.add_argument("--out", type=str, default="reports/week9_summary.csv", help="Output summary CSV")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(args.glob):
        try:
            rows.append(load_last_epoch(path))
        except Exception as e:
            print(f"[warn] skip {path}: {e}")

    if not rows:
        print("No files matched.")
        return

    df = pd.DataFrame(rows)

    # Derived metrics
    df["acc_drop_pp"] = (df["acc1"] - df["acc1_corrupt"]) * 100.0
    df["rel_robust_acc"] = df["acc1_corrupt"] / df["acc1"]  # higher is better
    df["ece_delta"] = df["ece_corrupt"] - df["ece"]
    df["MParams_tuned"] = df["trainable_params"] / 1e6

    # Light-weight robustness score (higher better): balance accuracy & calibration under corr
    # R = acc1_corrupt * exp(-2 * ece_corrupt) — tweak factor 2 if you want
    import numpy as np
    df["R_score"] = df["acc1_corrupt"] * np.exp(-2.0 * df["ece_corrupt"])

    keep_cols = [
        "source_file","dataset","model","method","policy","attn_only",
        "trainable_params","MParams_tuned","avg_step_time_s",
        "acc1","ece","acc1_corrupt","ece_corrupt","acc_drop_pp","rel_robust_acc","ece_delta",
        "noise_train","noise_p","label_noise","noise_gauss_std","noise_sp_amount","noise_erase_frac",
        "noise_blur_sigma","mixup_alpha","cutmix_alpha","fgsm_eps",
        "lr","lr_lora","lr_norm","weight_decay",
        "R_score"
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = float("nan") if df.select_dtypes(include=["number"]).shape[1] else "n/a"

    df = df[keep_cols]

    # Sort: best robustness first (acc1_corrupt desc, then lower ece_corrupt, then smaller acc_drop)
    df = df.sort_values(by=["acc1_corrupt","ece_corrupt","acc_drop_pp"], ascending=[False, True, True])

    # Pretty print to console
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved summary -> {args.out}")

if __name__ == "__main__":
    main()
