import argparse, csv, time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.week1.datasets import build_cifar10
from src.week1.vit_model import build_vit
from src.week1.utils import load_config, get_device, count_trainable_params, accuracy, expected_calibration_error

import platform, torch

def get_runtime_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")

    if platform.system() == "Windows":
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            pass
    return torch.device("cpu")

device = get_runtime_device(force_cpu=False)
print(f"Using device: {device}")


def build_cifar100(batch_size: int, num_workers: int, img_size: int = 224):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train = datasets.CIFAR100(root="./data", train=True, download=True, transform=tfm_train)
    test  = datasets.CIFAR100(root="./data", train=False, download=True, transform=tfm_test)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(test,  batch_size=max(256, batch_size), shuffle=False, num_workers=num_workers, pin_memory=True),
    )

def train_one_epoch(model, loader, optimizer, device, max_batches=0):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running_loss, running_acc, seen = 0.0, 0.0, 0
    for bi, (images, targets) in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches > 0 and bi >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        b = targets.size(0)
        running_loss += loss.item() * b
        running_acc  += accuracy(logits, targets) * b
        seen += b
    n = max(1, seen)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, total_ece, seen = 0.0, 0.0, 0.0, 0
    for bi, (images, targets) in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches > 0 and bi >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = loss_fn(logits, targets)
        b = targets.size(0)
        total_loss += loss.item() * b
        total_acc  += accuracy(logits, targets) * b
        total_ece  += expected_calibration_error(logits, targets) * b
        seen += b
    n = max(1, seen)
    return total_loss / n, total_acc / n, total_ece / n

    def enable_classifier_head(model: nn.Module):
    """
    Re-enable training for the classifier head like Week-3 did.
    """
    for attr in ("head", "fc", "classifier"):
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, choices=['cifar10','cifar100'], required=True)
    ap.add_argument('--backbone', type=str, default='vit_small_patch16_224')
    ap.add_argument('--method', type=str, choices=['norm','lora','hybrid'], required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3, help='base LR (head/norm)')
    ap.add_argument('--lora-lr', type=float, default=None, help='optional LR for LoRA params (hybrid)')
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--optimizer', type=str, default='adamw')
    ap.add_argument('--max-train-batches', type=int, default=0)
    ap.add_argument('--max-eval-batches', type=int, default=0)
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lora-alpha', type=int, default=16)
    ap.add_argument('--lora-drop', type=float, default=0.0)
    ap.add_argument('--out-csv', type=str, default='reports/results_week2.csv')
    args = ap.parse_args()

    device = get_device()
    print(f"\nUsing device: {device}")

    # Data
    if args.dataset == 'cifar10':
        train_loader, test_loader = build_cifar10(batch_size=args.batch_size, num_workers=args.num_workers)
        num_classes = 10
    else:
        train_loader, test_loader = build_cifar100(batch_size=args.batch_size, num_workers=args.num_workers)
        num_classes = 100

    # Model 
    model = build_vit(
        method=args.method,
        pretrained=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_drop,
        head_trainable=True,
        backbone=args.backbone,
    ).to(device)
    
    if args.unfreeze:
    unfreeze_classifier_head(model)

    import torch.nn as nn

    def _set_num_classes(m, num_classes):
        if hasattr(m, "reset_classifier"):
            m.reset_classifier(num_classes=num_classes)
            return
        
        if hasattr(m, "head") and isinstance(m.head, nn.Linear):
            in_f = m.head.in_features
            m.head = nn.Linear(in_f, num_classes)
            return
        
        if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
            in_f = m.fc.in_features
            m.fc = nn.Linear(in_f, num_classes)
            return
        
        if hasattr(m, "classifier") and isinstance(m.classifier, nn.Linear):
            in_f = m.classifier.in_features
            m.classifier = nn.Linear(in_f, num_classes)
            return
        raise RuntimeError("Could not set classifier layer for num_classes.")

    _set_num_classes(model, num_classes)

    model = model.to(device)
    head = getattr(model, "head", None) or getattr(model, "fc", None) or getattr(model, "classifier", None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True

    # Count trainables
    trn, tot, pct = count_trainable_params(model)
    print(f"Backbone: {args.backbone}")
    print(f"Trainable params: {trn} / {tot} ({pct:.3f}%)")

    # Optimizer with optional decoupled LR for Hybrid
    params_head, params_norm, params_lora = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        ln = n.lower()
        if 'norm' in ln or 'layernorm' in ln:
            params_norm.append(p)
        elif 'lora_' in ln or ln.endswith('.a') or ln.endswith('.b') or 'lora' in ln:
            params_lora.append(p)
        else:
            params_head.append(p)

    if args.method == 'hybrid' and args.lora_lr is not None:
        param_groups = [
            {'params': params_head, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': params_norm, 'lr': args.lr, 'weight_decay': 0.0},
            {'params': params_lora, 'lr': args.lora_lr, 'weight_decay': 0.0},
        ]
        print(f"LRs: head/norm={args.lr}, lora={args.lora_lr}")
    elif args.method == 'lora':
        # same LR for LoRA + head 
        param_groups = [{'params': params_head + params_lora, 'lr': args.lr, 'weight_decay': 0.0}]
        print(f"LRs: lora/head={args.lr}")
    else:
        # norm or hybrid without decoupled lora LR
        param_groups = []
        if params_head: param_groups.append({'params': params_head, 'lr': args.lr, 'weight_decay': args.weight_decay})
        if params_norm: param_groups.append({'params': params_norm, 'lr': args.lr, 'weight_decay': 0.0})
        if params_lora: param_groups.append({'params': params_lora, 'lr': (args.lora_lr or args.lr), 'weight_decay': 0.0})
        print(f"LRs: base={args.lr}" + (f", lora={args.lora_lr}" if args.lora_lr else ""))

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(param_groups)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    best_acc, t0 = 0.0, time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, max_batches=args.max_train_batches)
        te_loss, te_acc, te_ece = evaluate(model, test_loader, device, max_batches=args.max_eval_batches)
        scheduler.step()
        print(f"train loss {tr_loss:.4f} | train acc {tr_acc:.4f} | val loss {te_loss:.4f} | val acc {te_acc:.4f} | ECE {te_ece:.4f}")
        best_acc = max(best_acc, te_acc)

    total_time = time.time() - t0

    # Log
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ['ts','method','dataset','backbone','tuned_params_%','acc@1','ece','train_time_s','epochs']
    row = [int(time.time()), args.method, args.dataset, args.backbone, f"{pct:.4f}",
           f"{te_acc:.4f}", f"{te_ece:.4f}", f"{total_time:.1f}", args.epochs]
    write_header = not out_csv.exists()
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)

    # Save trainable-only checkpoint
    ckpt_path = Path('reports') / f"ckpt_{args.method}.pt"
    trainable_state = {k: v for k, v in model.state_dict().items()
                       if not hasattr(v, 'requires_grad') or v.requires_grad}
    torch.save({'config': vars(args), 'trainable_state': trainable_state}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

if __name__ == '__main__':
    main()
