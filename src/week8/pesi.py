# scripts/pesi_score.py
import argparse, csv, math, statistics as stats
from collections import defaultdict

ALPHA = 2.0  # calibration penalty weight

def normalize_column(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None,"")]
    med = stats.median(vals) if vals else 1.0
    for r in rows:
        if r.get(key) not in (None,""):
            r[f"{key}_norm"] = float(r[key]) / med if med != 0 else float(r[key])
        else:
            r[f"{key}_norm"] = ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_csv", required=True, help="CSV that already merged ID/C and latency + meta (one row per run)")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # Read
    rows = []
    with open(args.merged_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Normalize efficiency axes
    normalize_column(rows, "trainable_params_pct")
    normalize_column(rows, "latency_ms_cpu")

    # PESI
    for r in rows:
        try:
            acc_id = float(r["acc_id"])
            acc_c = float(r["acc_c_mean"])
            ece = float(r["ece_id"])
            p = float(r["trainable_params_pct_norm"])
            lat = float(r["latency_ms_cpu_norm"])
            utility = 0.5 * (acc_id + acc_c)
            efficiency = 1.0 / math.sqrt(max(p, 1e-12) * max(lat, 1e-12))
            calib = math.exp(-ALPHA * ece)
            r["pesi"] = utility * efficiency * calib
        except:
            r["pesi"] = ""
    # Write
    fields = rows[0].keys()
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] Wrote PESI to {args.out_csv}")

if __name__ == "__main__":
    main()
