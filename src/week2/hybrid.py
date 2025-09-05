import argparse, itertools, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['cifar10','cifar100'])
    ap.add_argument('--backbone', default='vit_small_patch16_224')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lora_lrs', type=str, default='5e-4,1e-3,2e-3')
    ap.add_argument('--norm_lrs', type=str, default='5e-4,1e-3,2e-3')
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_alpha', type=int, default=16)
    ap.add_argument('--lora_drop', type=float, default=0.0)
    ap.add_argument('--out_csv', type=str, default='reports/results_week2.csv')
    ap.add_argument('--max-train-batches', type=int, default=0)
    ap.add_argument('--max-eval-batches', type=int, default=0)
    args = ap.parse_args()

    lora_lrs = [float(x) for x in args.lora_lrs.split(',')]
    norm_lrs = [float(x) for x in args.norm_lrs.split(',')]

    repo_root = Path(__file__).resolve().parents[2]  # points to your project root (cv/)
    for l_lr, n_lr in itertools.product(lora_lrs, norm_lrs):
        cmd = [
            sys.executable, "-m", "src.week2.train",
            "--dataset", args.dataset,
            "--backbone", args.backbone,
            "--method", "hybrid",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--lr", str(n_lr),
            "--lora-lr", str(l_lr),
            "--lora-r", str(args.lora_r),
            "--lora-alpha", str(args.lora_alpha),
            "--lora-drop", str(args.lora_drop),
            "--out-csv", args.out_csv,
            "--max-train-batches", str(args.max_train_batches),
            "--max-eval-batches", str(args.max_eval_batches),
        ]
        print(">>>", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(repo_root))

if __name__ == "__main__":
    main()
