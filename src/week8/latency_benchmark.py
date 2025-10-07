# scripts/latency_benchmark.py
import argparse, csv, os, time, json
from pathlib import Path
import torch

def load_model(ckpt, device):
    obj = torch.load(ckpt, map_location=device)
    if hasattr(obj, "state_dict"):
        model = obj
        model.to(device)
        model.eval()
        return model
    raise RuntimeError("Save a whole-model checkpoint via torch.save(model, path).")

@torch.no_grad()
def time_inference(model, device, shape=(1,3,224,224), warmup=10, iters=100):
    x = torch.randn(*shape, device=device)
    # warmups
    if device == "cuda":
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
    else:
        for _ in range(warmup):
            _ = model(x)
        t0 = time.time()
        for _ in range(iters):
            _ = model(x)
        dt = time.time() - t0
    return (dt / iters) * 1000.0  # ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tag_json", type=str, default="{}")
    args = ap.parse_args()

    # GPU if available
    dev_gpu = "cuda" if torch.cuda.is_available() else None
    dev_cpu = "cpu"

    res = {}
    if dev_gpu:
        model_gpu = load_model(args.ckpt, dev_gpu)
        ms_gpu = time_inference(model_gpu, dev_gpu, (1,3,args.img_size,args.img_size), args.warmup, args.iters)
        res["latency_ms_gpu"] = ms_gpu
    model_cpu = load_model(args.ckpt, dev_cpu)
    ms_cpu = time_inference(model_cpu, dev_cpu, (1,3,args.img_size,args.img_size), args.warmup, args.iters)
    res["latency_ms_cpu"] = ms_cpu

    meta = json.loads(args.tag_json)
    fieldnames = ["ckpt","latency_ms_cpu","latency_ms_gpu"] + sorted(list(meta.keys()))
    Path(os.path.dirname(args.out_csv)).mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header: w.writeheader()
        row = {"ckpt": args.ckpt}
        row.update(res)
        row.update(meta)
        w.writerow(row)
    print(f"[OK] Latency CPU={res.get('latency_ms_cpu','-'):.3f} ms | GPU={res.get('latency_ms_gpu','-')} ms")

if __name__ == "__main__":
    main()
