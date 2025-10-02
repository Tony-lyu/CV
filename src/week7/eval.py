import argparse, csv, json, math, os, re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def minmax_norm(vals):
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    if not clean: return lambda x: float("nan")
    lo, hi = min(clean), max(clean)
    if hi <= lo: return lambda x: 0.0
    return lambda x: (x - lo) / (hi - lo)

def edge_score(acc_pct, ece, trainable_pct_01, step_time_norm, mem_norm,
               lam1, lam2, lam3, lam4):
    # acc_pct is 0..100; others are 0..1
    return acc_pct - lam1*(ece*100.0) - lam2*(trainable_pct_01*100.0) - lam3*(step_time_norm*100.0) - lam4*(mem_norm*100.0)

def pareto_front(points):
    # Maximize acc, minimize pct (primary 2D). Tie-breakers ignored for simplicity.
    front = []
    for i,a in enumerate(points):
        dominated = False
        for j,b in enumerate(points):
            if i == j: continue
            if (b["acc"] >= a["acc"] and b["pct"] <= a["pct"]) and (b["acc"] > a["acc"] or b["pct"] < a["pct"]):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front

def load_rows_with_source(paths):
    items = []  # (src_path, row_dict)
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for fn in p.glob("*.csv"):
                items += load_rows_with_source([fn])
            continue
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            for d in r:
                items.append((str(p), d))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="CSV files or directories (you can pass multiple).")
    ap.add_argument("--total_params", type=float, required=True, help="Backbone total parameter count.")
    ap.add_argument("--out_dir", default="reports/week7", help="Output dir for plots/tables.")
    ap.add_argument("--lam", nargs=4, type=float, default=[1.5, 2.0, 1.0, 0.5], help="EdgeScore lambdas: ECE, Trainable%, StepTime, PeakMem")
    ap.add_argument("--filter_dataset", default="cifar100")
    ap.add_argument("--filter_model", default="vit_small_patch16_224")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    items = load_rows_with_source(args.logs)

    # Keep last epoch per *file*
    last_by_src = {}
    for src, d in items:
        if d.get("dataset","") != args.filter_dataset: continue
        if d.get("model","")   != args.filter_model:   continue
        try:
            ep = int(float(d.get("epoch", 0)))
        except: ep = 0
        if (src not in last_by_src) or (ep > last_by_src[src].get("_epoch", -1)):
            # normalize fields
            acc = float(d.get("acc1","nan"))
            if acc <= 1.0: acc *= 100.0  # convert 0..1 to %
            ece = float(d.get("ece","nan"))
            trp = float(d.get("trainable_params","0"))
            time_s = float(d.get("avg_step_time_s","nan"))
            mem_b  = float(d.get("peak_mem_bytes","nan"))
            last_by_src[src] = dict(
                _epoch=ep,
                method=d.get("method",""),
                policy=d.get("policy","n.a."),
                acc=acc, ece=ece,
                trainable_params=trp,
                time_s=time_s, mem_b=mem_b
            )

    recs = []
    for src, r in last_by_src.items():
        pct01 = (r["trainable_params"] / args.total_params) if args.total_params else float("nan")
        recs.append({
            "src": src,
            "label": re.sub(r"_s\d+\.csv$", "", Path(src).name),  # e.g., budget0p5_lora_only
            "method": r["method"],
            "policy": r["policy"],
            "acc": r["acc"],
            "ece": r["ece"],
            "pct": pct01 * 100.0,
            "pct01": pct01,
            "time": r["time_s"],
            "mem": r["mem_b"],
        })

    if not recs:
        print("No matching rows found.")
        return

    # Normalizers for EdgeScore
    st_norm = minmax_norm([r["time"] for r in recs if not math.isnan(r["time"])])
    mem_norm = minmax_norm([r["mem"]  for r in recs if not math.isnan(r["mem"])])

    pts = []
    for i, r in enumerate(recs):
        es = edge_score(r["acc"], r["ece"], r["pct01"],
                        0.0 if math.isnan(r["time"]) else st_norm(r["time"]),
                        0.0 if math.isnan(r["mem"])  else mem_norm(r["mem"]),
                        *args.lam)
        pts.append({**r, "idx": i, "edgescore": es})

    # Pareto front
    front = pareto_front(pts)
    front_srcs = {p["src"] for p in front}

    # Save table
    out_csv = Path(args.out_dir) / "week7_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["src","label","method","policy","acc","ece","pct","time","mem","edgescore"])
        w.writeheader()
        for p in pts:
            w.writerow({k:p[k] for k in w.fieldnames})

    # Plot: Acc vs Trainable% (bubble = ECE)
    methods = sorted(set(p["method"] for p in pts))
    color_map = {m: plt.cm.tab10(i % 10) for i,m in enumerate(methods)}
    plt.figure(figsize=(8,6))
    for p in pts:
        edge = "black" if p["src"] in front_srcs else "none"
        ece_bubble = 200*max(0.05, (p["ece"] if not math.isnan(p["ece"]) else 0.05))
        plt.scatter(p["pct"], p["acc"], s=ece_bubble, alpha=0.7,
                    facecolors=color_map[p["method"]], edgecolors=edge, linewidths=1.3)
    plt.xlabel("Trainable params (%)")
    plt.ylabel("Top-1 Acc (%)")
    plt.title("Pareto: Acc vs Trainable% (bubble = ECE)")
    # legend by method
    handles = [plt.Line2D([0],[0], marker='o', color='w', label=m,
                           markerfacecolor=color_map[m], markersize=9)
               for m in methods]
    plt.legend(handles=handles, title="method", loc="lower right")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "week7_pareto.png", dpi=200)

    # EdgeScore bars — average by filename label (averages seeds)
    by_label = defaultdict(list)
    for p in pts:
        by_label[p["label"]].append(p["edgescore"])

    labels = sorted(by_label.keys())
    means = [sum(by_label[l]) / len(by_label[l]) for l in labels]

    plt.figure(figsize=(max(8, 0.25*len(labels)+6), 5))
    plt.bar(range(len(labels)), means)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("EdgeScore (higher is better)")
    plt.title("EdgeScore by configuration (mean over seeds)")
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "week7_edgescore.png", dpi=200)

    verdict = {
        "pareto_front_sources": sorted(list(front_srcs)),
        "note": "A hybrid is 'additive' for Q1 if any hybrid source appears on the Pareto front.",
    }
    with open(Path(args.out_dir) / "week7_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

if __name__ == "__main__":
    main()
