# tools/eval/week7_eval.py
import argparse, csv, json, math, os
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt

Row = namedtuple("Row", [
    "epoch","dataset","model","method","policy","attn_only",
    "lora_r","lora_alpha","lora_dropout","lr","lr_lora","lr_norm",
    "weight_decay","clip_grad","trainable_params","avg_step_time_s",
    "peak_mem_bytes","acc1","ece","grad_norm_lora_mean","grad_norm_norm_mean",
    "update_norm_lora_mean","update_norm_norm_mean","cosine_ln_mean","cosine_ln_p90",
    "freeze_lora_steps","freeze_norm_steps","alt_freeze_every","alt_freeze_order","seed","tag"
])

def parse_float(x):
    try:
        if x in ("", "nan", "None", None): return float("nan")
        return float(x)
    except:
        return float("nan")

def parse_int(x):
    try:
        if x in ("", "nan", "None", None): return 0
        return int(float(x))
    except:
        return 0

def load_csvs(paths):
    rows = []
    for p in paths:
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.endswith(".csv"):
                    rows += load_csvs([os.path.join(p, fn)])
            continue
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            for d in r:
                rows.append(Row(
                    parse_int(d.get("epoch", 0)),
                    d.get("dataset",""),
                    d.get("model",""),
                    d.get("method",""),
                    d.get("policy",""),
                    parse_int(d.get("attn_only",0)),
                    parse_float(d.get("lora_r", "nan")),
                    parse_float(d.get("lora_alpha", "nan")),
                    parse_float(d.get("lora_dropout","nan")),
                    parse_float(d.get("lr","nan")),
                    parse_float(d.get("lr_lora","nan")),
                    parse_float(d.get("lr_norm","nan")),
                    parse_float(d.get("weight_decay","nan")),
                    parse_float(d.get("clip_grad","nan")),
                    parse_int(d.get("trainable_params",0)),
                    parse_float(d.get("avg_step_time_s","nan")),
                    parse_float(d.get("peak_mem_bytes","nan")),
                    parse_float(d.get("acc1","nan")),
                    parse_float(d.get("ece","nan")),
                    parse_float(d.get("grad_norm_lora_mean","nan")),
                    parse_float(d.get("grad_norm_norm_mean","nan")),
                    parse_float(d.get("update_norm_lora_mean","nan")),
                    parse_float(d.get("update_norm_norm_mean","nan")),
                    parse_float(d.get("cosine_ln_mean","nan")),
                    parse_float(d.get("cosine_ln_p90","nan")),
                    parse_int(d.get("freeze_lora_steps",0)),
                    parse_int(d.get("freeze_norm_steps",0)),
                    parse_int(d.get("alt_freeze_every",0)),
                    d.get("alt_freeze_order",""),
                    parse_int(d.get("seed", d.get("run_seed", 0))),
                    d.get("tag", d.get("run_tag",""))
                ))
    return rows

def pct_trainable(rows):
    # Assumes your CSV has absolute trainable_params; we convert to %
    # Pass total_params via CLI to compute %, else just return absolute.
    pass

def edge_score(acc, ece, trainable_pct, step_time, peak_mem,
               lam1, lam2, lam3, lam4):
    # All terms expected in [0,1] scale except acc (%). We'll normalize internally.
    # acc as [0..100], others as z in [0..1].
    # EdgeScore = acc - lam1*ece*100 - lam2*trainable_pct*100 - lam3*step_time_norm*100 - lam4*mem_norm*100
    return acc - lam1*ece*100.0 - lam2*trainable_pct*100.0 - lam3*step_time*100.0 - lam4*peak_mem*100.0

def minmax_norm(vals):
    clean = [v for v in vals if not math.isnan(v)]
    if not clean:
        return lambda x: float("nan")
    lo, hi = min(clean), max(clean)
    if hi <= lo: return lambda x: 0.0
    return lambda x: (x - lo) / (hi - lo)

def pareto_front(points):
    # points: list of dict with keys "acc","ece","pct","time","mem","idx"
    # We’ll use 2D primary: maximize acc, minimize pct. Tie-breakers by ece, time, mem.
    # A dominates B if acc>= and pct<= (and at least one strict), with non-worse ece/time/mem if equal on primary.
    front = []
    for i,a in enumerate(points):
        dominated = False
        for j,b in enumerate(points):
            if i==j: continue
            primary = (b["acc"] >= a["acc"] and b["pct"] <= a["pct"]) and (b["acc"]>a["acc"] or b["pct"]<a["pct"])
            tie_ok = True
            if primary:
                # if primary dominates, mark dominated regardless of tie-breakers
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="CSV file(s) or directory(ies) with your aggregated logs")
    ap.add_argument("--total_params", type=float, required=True, help="Total parameter count of the backbone (for % calc)")
    ap.add_argument("--out_dir", default="reports/week7", help="Where to write plots/tables")
    ap.add_argument("--lam", nargs=4, type=float, default=[1.0, 1.0, 1.0, 1.0],
                    help="Lambdas for EdgeScore: lam1*ECE + lam2*Trainable% + lam3*StepTime + lam4*PeakMem")
    ap.add_argument("--filter_dataset", default="cifar100")
    ap.add_argument("--filter_model", default="vit_small_patch16_224")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_csvs(args.logs)
    rows = [r for r in rows if r.dataset==args.filter_dataset and r.model==args.filter_model]

    # Aggregate last-epoch metrics per (method, policy, tag, seed)
    key = lambda r: (r.method, r.policy, r.tag, r.seed)
    latest = {}
    for r in rows:
        k = key(r)
        if k not in latest or r.epoch > latest[k].epoch:
            latest[k] = r
    recs = list(latest.values())

    # Compute %-trainable using total params
    pts = []
    step_times = [r.avg_step_time_s for r in recs if not math.isnan(r.avg_step_time_s)]
    mems = [r.peak_mem_bytes for r in recs if not math.isnan(r.peak_mem_bytes)]
    st_norm = minmax_norm(step_times)
    mem_norm = minmax_norm(mems)

    table = []
    for idx, r in enumerate(recs):
        pct = (r.trainable_params / args.total_params) if args.total_params else float("nan")
        time_n = st_norm(r.avg_step_time_s) if not math.isnan(r.avg_step_time_s) else float("nan")
        mem_n  = mem_norm(r.peak_mem_bytes) if not math.isnan(r.peak_mem_bytes) else float("nan")
        es = edge_score(r.acc1*100 if r.acc1<=1.0 else r.acc1,  # accept 0..1 or 0..100
                        r.ece,
                        pct,
                        time_n if not math.isnan(time_n) else 0.0,
                        mem_n if not math.isnan(mem_n) else 0.0,
                        *args.lam)
        pts.append({
            "idx": idx, "acc": (r.acc1*100 if r.acc1<=1.0 else r.acc1),
            "pct": pct*100, "ece": r.ece, "time": r.avg_step_time_s, "mem": r.peak_mem_bytes,
            "method": r.method, "policy": r.policy, "tag": r.tag, "seed": r.seed, "edgescore": es
        })
        table.append({
            "method": r.method, "policy": r.policy, "tag": r.tag, "seed": r.seed,
            "acc": (r.acc1*100 if r.acc1<=1.0 else r.acc1),
            "ece": r.ece, "trainable_pct": pct*100,
            "avg_step_time_s": r.avg_step_time_s, "peak_mem_bytes": r.peak_mem_bytes,
            "edgescore": es
        })

    # Pareto front (Acc vs Trainable%)
    front = pareto_front(pts)
    front_set = {(p["method"], p["policy"], p["tag"], p["seed"]) for p in front}

    # Save table
    out_csv = os.path.join(args.out_dir, "week7_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        for t in table:
            w.writerow(t)

    # Plot Acc vs Trainable%, bubble size=ECE, color by method
    methods = sorted(set(p["method"] for p in pts))
    color_map = {m: plt.cm.tab10(i % 10) for i,m in enumerate(methods)}
    plt.figure(figsize=(8,6))
    for p in pts:
        edge = "black" if (p["method"],p["policy"],p["tag"],p["seed"]) in front_set else "none"
        plt.scatter(p["pct"], p["acc"], s=200*max(0.05, p["ece"]), alpha=0.7,
                    facecolors=color_map[p["method"]], edgecolors=edge, linewidths=1.5, label=p["method"])
    plt.xlabel("Trainable params (%)"); plt.ylabel("Top-1 Acc (%)")
    plt.title("Pareto: Acc vs Trainable% (bubble = ECE)")
    # Unique legend
    handles = []
    seen = set()
    for m in methods:
        if m not in seen:
            seen.add(m)
            handles.append(plt.Line2D([0],[0], marker='o', color='w', label=m,
                                      markerfacecolor=color_map[m], markersize=10))
    plt.legend(handles=handles, title="method", loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "week7_pareto.png"), dpi=200)

    # EdgeScore bars (group by tag)
    # Expect tags like: budget0p2_lora_only, budget0p2_hybrid_50_50, etc.
    by_tag = defaultdict(list)
    for p in pts:
        by_tag[p["tag"]].append(p["edgescore"])
    tag_means = {t: sum(v)/len(v) for t,v in by_tag.items() if v}
    order = sorted(tag_means.keys(), key=lambda t: tag_means[t], reverse=True)
    plt.figure(figsize=(10,5))
    plt.bar([i for i,_ in enumerate(order)], [tag_means[t] for t in order])
    plt.xticks([i for i,_ in enumerate(order)], order, rotation=45, ha="right")
    plt.ylabel("EdgeScore (higher is better)")
    plt.title("EdgeScore by configuration (mean over seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "week7_edgescore.png"), dpi=200)

    # Final textual verdicts
    verdict = {
        "pareto_front_tags": sorted(list({p['tag'] for p in front})),
        "q1_answer": "Hybrid is additive iff any hybrid tag appears on the Pareto front; otherwise interference.",
        "q2_answer": "Use layer-sweep results (tags starting with layersweep_) to identify where LoRA vs Norm/EFFT help most."
    }
    with open(os.path.join(args.out_dir, "week7_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)

if __name__ == "__main__":
    main()
