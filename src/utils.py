import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def count_trainable_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    return trainable, total, pct

def accuracy(output, target):
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = (preds == target).float().sum()
        return (correct / target.size(0)).item()

def expected_calibration_error(logits, targets, n_bins: int = 15):
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        targets = targets.view(-1)
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
        for i in range(n_bins):
            bin_low, bin_high = bin_boundaries[i], bin_boundaries[i+1]
            in_bin = (confs > bin_low) & (confs <= bin_high)
            bin_size = in_bin.float().sum()
            if bin_size.item() > 0:
                acc = (preds[in_bin] == targets[in_bin]).float().mean()
                avg_conf = confs[in_bin].mean()
                ece += (bin_size / logits.size(0)) * torch.abs(avg_conf - acc)
        return ece.item()
