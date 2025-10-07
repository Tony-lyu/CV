# scripts/merge_metrics.py
import argparse, csv
from collections import defaultdict

def index_by_key(rows, key):
    idx = defaultdict(list)
    for r in rows:
        idx[r[key]].append(r)
    return idx

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", required=True)
    ap.add_argument("--latency_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--join_key", default="ckpt")
    args = ap.parse_args()

    eval_rows = read_csv(args.eval_csv)
    lat_rows = read_csv(args.latency_csv)
    idx_lat = index_by_key(lat_rows, args.join_key)

    merged = []
    for r in eval_rows:
        key = r[args.join_key]
        lat_matches = idx_lat.get(key, [{}])
        for lm in lat_matches:
            mr = dict(r)
            for k,v in lm.items():
                if k not in mr:
                    mr[k] = v
            merged.append(mr)

    fields = set()
    for r in merged:
        fields |= set(r.keys())
    fields = list(fields)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)
    print(f"[OK] merged rows={len(merged)} -> {args.out_csv}")

if __name__ == "__main__":
    main()
