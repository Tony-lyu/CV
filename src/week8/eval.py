# scripts/cifar_c_eval.py
import argparse, json, os, time, math, csv
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
    "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression"
]

def build_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    correct, total = 0, 0
    nll_sum = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        nll_sum += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    acc = correct / total
    nll = nll_sum / total
    return acc, nll

def expected_calibration_error(logits, targets, n_bins=15):
    # logits: [N, C], targets: [N], both torch tensors on CPU
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    ece = torch.zeros(1)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i+1])
        if mask.any():
            acc = (preds[mask] == targets[mask]).float().mean()
            conf = confs[mask].mean()
            ece += (mask.float().mean()) * torch.abs(conf - acc)
    return ece.item()

@torch.no_grad()
def eval_loader_with_ece(model, loader, device):
    model.eval()
    correct, total = 0, 0
    nll_sum = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    all_logits = []
    all_targets = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        nll_sum += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
    acc = correct / total
    nll = nll_sum / total
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    ece = expected_calibration_error(logits, targets)
    return acc, nll, ece

def load_model(ckpt_path, device):
    """
    Expect a torch checkpoint that contains:
      - 'model' (state_dict) OR full scripted
      - Optionally model definition saved alongside (torch.save(model))
    For safety we attempt both.
    """
    obj = torch.load(ckpt_path, map_location=device)
    if hasattr(obj, "state_dict"):  # entire model object
        model = obj
        model.to(device)
        return model
    if isinstance(obj, dict) and "model" in obj:
        # you must import/construct your model before loading.
        raise RuntimeError("Checkpoint contains a state_dict; please load in your model code before calling this script "
                           "or export a 'whole model' checkpoint with torch.save(model, path).")
    raise RuntimeError("Unsupported checkpoint format. Save whole model with torch.save(model, path).")

def cifar_c_dataset(root, corruption, severity, transform):
    images = np.load(os.path.join(root, f"{corruption}.npy"))
    labels = np.load(os.path.join(root, "labels.npy"))
    assert 1 <= severity <= 5
    N = images.shape[0] // 5
    sl = slice((severity-1)*N, severity*N)
    imgs = images[sl]
    labs = labels[sl]
    # Convert to dataset
    from PIL import Image
    class NPDataset(torch.utils.data.Dataset):
        def __len__(self): return len(labs)
        def __getitem__(self, idx):
            img = Image.fromarray(imgs[idx])
            return transform(img), int(labs[idx])
    return NPDataset()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--cifar_c_root", type=str, default=os.environ.get("CIFAR_C_DIR", "./CIFAR-100-C"))
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--tag_json", type=str, default="{}",
                    help="JSON string with meta info to include in CSV (e.g., method,budget_pct,seed,trainable_params_pct,latency_ms_cpu)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = build_transforms(args.img_size)

    # Load model (must be a whole-model checkpoint)
    model = load_model(args.ckpt, device)

    # ID eval (CIFAR-100)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    acc_id, nll_id, ece_id = eval_loader_with_ece(model, test_loader, device)

    # C eval (CIFAR-100-C)
    acc_c_list = []
    for corr in tqdm(CORRUPTIONS, desc="CIFAR-100-C"):
        acc_sev = []
        for s in range(1, 6):
            ds = cifar_c_dataset(args.cifar_c_root, corr, s, tfm)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            acc_s, _ = eval_loader(model, dl, device)
            acc_sev.append(acc_s)
        acc_c_list.append(np.mean(acc_sev))
    acc_c_mean = float(np.mean(acc_c_list))
    acc_c_worst = float(np.min(acc_c_list))

    # Write CSV row
    meta = json.loads(args.tag_json)
    fieldnames = [
        "ckpt","acc_id","ece_id","nll_id","acc_c_mean","acc_c_worst"
    ] + sorted(list(meta.keys()))
    Path(os.path.dirname(args.out_csv)).mkdir(parents=True, exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header: w.writeheader()
        row = {"ckpt": args.ckpt, "acc_id": acc_id, "ece_id": ece_id, "nll_id": nll_id,
               "acc_c_mean": acc_c_mean, "acc_c_worst": acc_c_worst}
        row.update(meta)
        w.writerow(row)
    print(f"[OK] ID acc={acc_id:.4f} ECE={ece_id:.4f} NLL={nll_id:.4f} | C mean={acc_c_mean:.4f} worst={acc_c_worst:.4f}")

if __name__ == "__main__":
    main()
