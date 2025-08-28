import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .datasets import build_cifar10
from .vit_model import build_vit
from .utils import load_config, get_device, count_trainable_params, accuracy, expected_calibration_error

def train_one_epoch(model, loader, optimizer, device, max_batches = 0):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running_loss, running_acc, seen = 0.0, 0.0, 0
    for bi, (images, targets) in enumerate(tqdm(loader, desc='train', leave=False)):
        if max_batches > 0 and bi >= max_batches:
            break
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        batch = targets.size(0)
        running_loss += loss.item() * batch
        running_acc += accuracy(logits, targets) * batch
        seen += batch
    n = max(1, seen)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device, max_batches = 0):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, total_ece, seen = 0.0, 0.0, 0.0, 0
    for bi, (images, targets) in enumerate(tqdm(loader, desc='eval', leave=False)):
        if max_batches > 0 and bi >= max_batches:
            break
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = loss_fn(logits, targets)
        batch = targets.size(0)
        total_loss += loss.item() * batch
        total_acc += accuracy(logits, targets) * batch
        total_ece += expected_calibration_error(logits, targets) * batch
        seen += batch
    n = max(1, seen)
    return total_loss / n, total_acc / n, total_ece / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='Path to YAML config')
    args = ap.parse_args()

    cfg = load_config(args.config)
    backbone = cfg.get('model', {}).get('backbone', 'vit_base_patch16_224')
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = build_cifar10(batch_size=cfg['train']['batch_size'],
                                              num_workers=cfg['train']['num_workers'])

    # Model
    model = build_vit(method=cfg['method']['name'],
                      pretrained=True,
                      lora_r=cfg['method'].get('lora_r', 8),
                      lora_alpha=cfg['method'].get('lora_alpha', 16),
                      lora_dropout=cfg['method'].get('lora_dropout', 0.05),
                      head_trainable=cfg['method'].get('head_trainable', True),
                      backbone=backbone)
    model.to(device)

    trn, tot, pct = count_trainable_params(model)
    print(f"Backbone: {backbone}")
    print(f"Trainable params: {trn} / {tot} ({pct:.3f}%)")

    # Optimizer & sched
    optim_name = str(cfg['train'].get('optimizer', 'adamw').lower())
    lr = float(cfg['train']['lr'])
    wd = float(cfg['train']['weight_decay'])
    epochs = int(cfg['train']['epochs'])
    max_train_batches = int(cfg['train'].get('max_train_batches', 0))
    max_eval_batches = int(cfg['train'].get('max_eval_batches', 0))

    params = filter(lambda p: p.requires_grad, model.parameters())
    if optim_name == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Train loop
    best_acc = 0.0
    start_time = time.time()
    for epoch in range(1, cfg['train']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, max_batches=max_train_batches)
        te_loss, te_acc, te_ece = evaluate(model, test_loader, device, max_batches=max_eval_batches)
        scheduler.step()
        print(f"train loss {tr_loss:.4f} | train acc {tr_acc:.4f} | val loss {te_loss:.4f} | val acc {te_acc:.4f} | ECE {te_ece:.4f}")
        best_acc = max(best_acc, te_acc)

    total_time = time.time() - start_time

    # Logging
    out_csv = Path('reports/results_week1.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ['ts', 'method', 'dataset', 'backbone', 'tuned_params_%', 'acc@1', 'ece', 'train_time_s', 'epochs']
    row = [int(time.time()), cfg['method']['name'], 'cifar10', 'vit_base_patch16_224', f"{pct:.4f}", f"{te_acc:.4f}", f"{te_ece:.4f}", f"{total_time:.1f}", cfg['train']['epochs']]
    write_header = not out_csv.exists()
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    # Save a lightweight checkpoint (trainable params only)
    ckpt_path = Path('reports') / f"ckpt_{cfg['method']['name']}.pt"
    trainable_state = {k: v for k, v in model.state_dict().items() if not hasattr(v, 'requires_grad') or v.requires_grad}
    torch.save({'config': cfg, 'trainable_state': trainable_state}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

if __name__ == '__main__':
    main()
