import os, csv, time, math, argparse
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import timm
from tqdm import tqdm

from hybrid_layer import build_method_vit, count_trainable_params

# ---------------------------
# Metrics
# ---------------------------
@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

@torch.no_grad()
def expected_calibration_error(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    accs = (preds == targets).float()
    bins = torch.linspace(0, 1, steps=n_bins + 1, device=logits.device)
    ece = torch.zeros([], device=logits.device)
    for i in range(n_bins):
        # (0,1] style bins; include left-open to avoid double counting 0
        in_bin = (confs > bins[i]) & (confs <= bins[i + 1])
        prop = in_bin.float().mean()
        if prop > 0:
            acc_bin = accs[in_bin].float().mean()
            conf_bin = confs[in_bin].float().mean()
            ece = ece + torch.abs(conf_bin - acc_bin) * prop
    return float(ece.item())

# ---------------------------
# Diagnostics
# ---------------------------
def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    na = a.norm().item(); nb = b.norm().item()
    if na < eps or nb < eps: return float("nan")
    return float((a @ b / (na * nb)).item())

class GradMonitor:
    """
    Per-layer grad/update norms for LoRA vs Norm + cosine similarity,
    and per-step global grad-norm EMAs for LR parity.
    """
    def __init__(self, model: nn.Module, ema_beta: float = 0.9):
        self.model = model
        self.grad_norms = defaultdict(lambda: {"lora":0.0,"norm":0.0})
        self.update_norms = defaultdict(lambda: {"lora":0.0,"norm":0.0})
        self.cosines = {}
        self._pre_step = {}
        # EMAs for dynamic LR parity (group-level)
        self.ema_beta = ema_beta
        self.ema_grad_lora = None
        self.ema_grad_norm = None
        # latest per-step totals
        self.last_grad_lora = 0.0
        self.last_grad_norm = 0.0

    def zero_epoch(self):
        self.grad_norms.clear(); self.update_norms.clear(); self.cosines.clear(); self._pre_step.clear()
        self.last_grad_lora = 0.0; self.last_grad_norm = 0.0

    def _layer_of(self, name: str) -> str:
        parts = name.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]

    def capture_pre_step(self):
        self._pre_step.clear()
        for idx, (n, p) in enumerate(self.model.named_parameters()):
            if p.requires_grad:
                self._pre_step[idx] = (n, p.data.detach().clone())

    def after_backward(self):
        grads_l = defaultdict(list); grads_n = defaultdict(list)
        dev = next(self.model.parameters()).device
        # aggregate per-parameter grads
        for name, p in self.model.named_parameters():
            if (not p.requires_grad) or (p.grad is None): continue
            g = p.grad.detach().flatten()
            layer = self._layer_of(name)
            ln = name.lower()
            is_lora = ("lora_" in ln) or ("lora" in ln) or ln.endswith(".a") or ln.endswith(".b")
            is_norm = (("norm" in ln) or ("layernorm" in ln) or ("ln" in ln)) and (("weight" in ln) or ("bias" in ln))
            if is_lora: grads_l[layer].append(g)
            if is_norm: grads_n[layer].append(g)

        # per-layer stats
        for layer in set(list(grads_l.keys()) + list(grads_n.keys())):
            gl = torch.cat(grads_l[layer]) if layer in grads_l else torch.zeros(1, device=dev)
            gn = torch.cat(grads_n[layer]) if layer in grads_n else torch.zeros(1, device=dev)
            self.grad_norms[layer]["lora"] = gl.norm().item()
            self.grad_norms[layer]["norm"] = gn.norm().item()
            samesize = min(gl.numel(), gn.numel())
            self.cosines[layer] = _cos(gl[:samesize], gn[:samesize]) if samesize > 0 else float("nan")

        # group-level totals (for LR parity)
        total_l = sum(v["lora"] for v in self.grad_norms.values() if math.isfinite(v["lora"]))
        total_n = sum(v["norm"] for v in self.grad_norms.values() if math.isfinite(v["norm"]))
        self.last_grad_lora = float(total_l)
        self.last_grad_norm = float(total_n)
        # EMA update
        if self.ema_grad_lora is None:
            self.ema_grad_lora = self.last_grad_lora
            self.ema_grad_norm = self.last_grad_norm
        else:
            b = self.ema_beta
            self.ema_grad_lora = b * self.ema_grad_lora + (1-b) * self.last_grad_lora
            self.ema_grad_norm = b * self.ema_grad_norm + (1-b) * self.last_grad_norm

    def after_step(self):
        deltas_l = defaultdict(list); deltas_n = defaultdict(list)
        for idx, (name, p) in enumerate(self.model.named_parameters()):
            if not p.requires_grad or idx not in self._pre_step: continue
            prev_name, prev = self._pre_step[idx]
            # name should be stable; if not, skip the update norm logging for safety
            if name != prev_name: continue
            d = (p.data - prev).detach().flatten()
            layer = self._layer_of(name)
            ln = name.lower()
            is_lora = ("lora_" in ln) or ("lora" in ln) or ln.endswith(".a") or ln.endswith(".b")
            is_norm = (("norm" in ln) or ("layernorm" in ln) or ("ln" in ln)) and (("weight" in ln) or ("bias" in ln))
            if is_lora: deltas_l[layer].append(d)
            if is_norm: deltas_n[layer].append(d)
        for layer in set(list(deltas_l.keys()) + list(deltas_n.keys())):
            dl = torch.cat(deltas_l[layer]) if layer in deltas_l else torch.zeros(1)
            dn = torch.cat(deltas_n[layer]) if layer in deltas_n else torch.zeros(1)
            self.update_norms[layer]["lora"] = dl.norm().item()
            self.update_norms[layer]["norm"] = dn.norm().item()

    def summarize_epoch(self):
        def _mean(xs):
            xs = [x for x in xs if math.isfinite(x)]
            return (sum(xs)/len(xs)) if xs else float("nan")
        def _p90(xs):
            xs = [x for x in xs if math.isfinite(x)]
            if not xs: return float("nan")
            xs = sorted(xs); return xs[int(0.9*(len(xs)-1))]
        l_g = [v["lora"] for v in self.grad_norms.values()]
        n_g = [v["norm"] for v in self.grad_norms.values()]
        l_u = [v["lora"] for v in self.update_norms.values()]
        n_u = [v["norm"] for v in self.update_norms.values()]
        cos = list(self.cosines.values())
        return {
            "grad_norm_lora_mean": _mean(l_g),
            "grad_norm_norm_mean": _mean(n_g),
            "update_norm_lora_mean": _mean(l_u),
            "update_norm_norm_mean": _mean(n_u),
            "cosine_ln_mean": _mean(cos),
            "cosine_ln_p90": _p90(cos),
        }

class FreezeController:
    """
    Batch-wise selective freezing of LoRA/Norm groups.
    - Warm-up: freeze_lora_steps / freeze_norm_steps
    - Alternation: alt_every batches, toggling which group is frozen
    """
    def __init__(self, model: nn.Module, freeze_lora_steps: int, freeze_norm_steps: int,
                 alt_every: int, alt_order: str):
        self.groups = collect_param_objects(model)
        # remember original trainable-ness for restoration
        for ps in (self.groups["lora"] + self.groups["norm"] + self.groups["other"]):
            if not hasattr(ps, "_orig_trainable"):
                ps._orig_trainable = bool(ps.requires_grad)
        self.step = 0
        self.freeze_lora_steps = int(max(0, freeze_lora_steps))
        self.freeze_norm_steps = int(max(0, freeze_norm_steps))
        self.alt_every = int(max(0, alt_every))
        self.alt_order = alt_order

    def _alt_freeze_flags(self):
        if self.alt_every <= 0:
            return False, False
        phase = (self.step // self.alt_every) % 2
        if self.alt_order == "lora-first":
            # phase 0: freeze lora, phase 1: freeze norm
            return phase == 0, phase == 1
        else:
            # norm-first
            return phase == 1, phase == 0

    def apply(self):
        # warm-ups
        freeze_lora = (self.step < self.freeze_lora_steps)
        freeze_norm = (self.step < self.freeze_norm_steps)

        # alternation (after warm-ups; union logic keeps a group frozen if any rule says so)
        alt_lora, alt_norm = self._alt_freeze_flags()
        if self.step >= max(self.freeze_lora_steps, self.freeze_norm_steps):
            freeze_lora = freeze_lora or alt_lora
            freeze_norm = freeze_norm or alt_norm

        # set requires_grad using original flags as baseline
        for p in self.groups["lora"]:
            p.requires_grad = p._orig_trainable and (not freeze_lora)
        for p in self.groups["norm"]:
            p.requires_grad = p._orig_trainable and (not freeze_norm)
        # leave "other" (including head) as originally configured
        self.step += 1

# ---------------------------
# Dynamic LR parity
# ---------------------------
class DynamicLRParity:
    """
    Balances LoRA vs Norm update strengths by scaling their LRs using EMA grad norms.

    Safeguards:
      - No adjustments while either group is frozen/absent
      - Cool-off for a few periods right after unfreeze (EMAs reset)
      - Deadband around target ratio (avoid needless twitching)
      - Limit consecutive scaling of the same group
      - Clamp absolute LRs to [min_lr, max_lr] after every change
    """
    def __init__(
        self,
        target_ratio=1.0,
        period=20,
        min_scale=0.5,
        max_scale=2.0,
        eps=1e-8,
        min_lr=1e-6,
        max_lr=1e-2,
        min_ema=1e-3,
        deadband_lo=2/3,     # no change if gL/gN is within [2/3, 3/2]
        deadband_hi=3/2,
        cooloff_periods=2,   # skip this many periods after groups become active
        max_consecutive=3,   # don't scale the same group more than N periods in a row
    ):
        self.target = target_ratio
        self.period = max(1, int(period))
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_consecutive = max_consecutive
        self.eps = eps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_ema = min_ema
        self.deadband_lo = deadband_lo

        self.deadband_hi = deadband_hi
        self.cooloff_periods = max(0, int(cooloff_periods))

        self._steps = 0
        self._prev_active = False
        self._cooloff_left = 0
        self._last_scaled = None        # "lora" | "norm" | None
        self._last_scaled_streak = 0

    def _both_groups_active(self, optimizer):
        names = set(pg.get("name") for pg in optimizer.param_groups)
        if not {"lora", "norm"}.issubset(names):
            return False
        active = {"lora": False, "norm": False}
        for pg in optimizer.param_groups:
            name = pg.get("name")
            if name in active:
                if any(p.requires_grad for p in pg["params"]):
                    active[name] = True
        return active["lora"] and active["norm"]

    def _clamp_lrs(self, optimizer):
        for pg in optimizer.param_groups:
            if "lr" in pg:
                pg["lr"] = float(max(self.min_lr, min(self.max_lr, pg["lr"])))

    def maybe_adjust(self, optimizer, monitor):
        self._steps += 1
        if (self._steps % self.period) != 0:
            return

        active_now = self._both_groups_active(optimizer)
        if not active_now:
            # mark inactive; next activation will reset EMAs & start cooloff
            self._prev_active = False
            return

        # transition: just became active → reset EMAs and start cooloff
        if active_now and not self._prev_active:
            if monitor is not None:
                monitor.ema_grad_lora = None
                monitor.ema_grad_norm = None
            self._prev_active = True
            self._cooloff_left = self.cooloff_periods
            self._last_scaled = None
            self._last_scaled_streak = 0
            return

        # honor cooloff window
        if self._cooloff_left > 0:
            self._cooloff_left -= 1
            return

        # need valid, non-trivial EMAs
        gL = monitor.ema_grad_lora if monitor is not None else None
        gN = monitor.ema_grad_norm if monitor is not None else None
        if gL is None or gN is None:
            return
        if gL < self.min_ema or gN < self.min_ema:
            return

        # ratio with guard
        ratio = gL / max(gN, self.eps)  # want ratio ≈ target
        # convert to "relative to target"
        ratio_rel = ratio / max(self.target, self.eps)

        # deadband: do nothing if already close enough
        if self.deadband_lo <= ratio_rel <= self.deadband_hi:
            return

        # determine which side to scale (shrink the dominant)
        scale_L = 1.0
        scale_N = 1.0
        scaled_group = None

        if ratio_rel > self.deadband_hi:
            # LoRA too strong, shrink LoRA
            raw = 1.0 / max(ratio_rel, self.eps)  # target is 1.0 after normalization
            scale_L = float(min(self.max_scale, max(self.min_scale, raw)))
            scaled_group = "lora"
        else:  # ratio_rel < deadband_lo → Norm too strong, shrink Norm
            raw = ratio_rel  # bring it up by shrinking Norm
            scale_N = float(min(self.max_scale, max(self.min_scale, raw)))
            scaled_group = "norm"

        # streak limiter: avoid shrinking the same side too many times in a row
        if scaled_group == self._last_scaled:
            if self._last_scaled_streak >= self.max_consecutive:
                return
            self._last_scaled_streak += 1
        else:
            self._last_scaled = scaled_group
            self._last_scaled_streak = 1

        # apply
        for pg in optimizer.param_groups:
            name = pg.get("name")
            if name == "lora":
                pg["lr"] *= scale_L
            elif name == "norm":
                pg["lr"] *= scale_N

        # keep LRs in a sane range
        self._clamp_lrs(optimizer)




# ---------------------------
# Data (augs per method)
# ---------------------------
def _build_transforms(dataset: str, img_size: int, mode: str):
    """
    mode = "week3"  -> Resize -> RandomCrop(pad=4) -> Flip -> ToTensor  (no Normalize)
    mode = "week4"  -> Resize -> Flip -> ToTensor -> Normalize
    """
    if dataset.lower() == "cifar100":
        mean = (0.5071, 0.4866, 0.4409); std = (0.2673, 0.2564, 0.2762)
    else:
        mean = (0.4914, 0.4822, 0.4465); std = (0.2023, 0.1994, 0.2010)

    if mode == "week3":
        train_tf = T.Compose([T.Resize((img_size,img_size)), T.RandomCrop(img_size, padding=4),
                              T.RandomHorizontalFlip(), T.ToTensor()])
        test_tf  = T.Compose([T.Resize((img_size,img_size)), T.ToTensor()])
    else:
        train_tf = T.Compose([T.Resize((img_size,img_size)), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize(mean, std)])
        test_tf  = T.Compose([T.Resize((img_size,img_size)), T.ToTensor(), T.Normalize(mean, std)])
    return train_tf, test_tf

def make_dataloaders(dataset: str, img_size: int, batch_size: int, workers: int, recipe_mode: str):
    train_tf, test_tf = _build_transforms(dataset, img_size, recipe_mode)
    if dataset.lower() == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(root="./data", train=True,  download=True, transform=train_tf)
        test_ds  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=test_tf)
        num_classes = 100
    elif dataset.lower() == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
        test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader, num_classes

# ---------------------------
# Param groups (+names for parity)
# ---------------------------
def split_param_groups_week3(model: nn.Module, base_lr: float, lora_lr: float, norm_lr: float, wd_other: float = 0.05):
    """Head/other: WD=0.05; Norm: WD=0; LoRA: WD=0. LRs as provided."""
    pg_lora, pg_norm, pg_other = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        ln = n.lower()
        if ("lora_" in ln) or ("lora" in ln) or ln.endswith(".a") or ln.endswith(".b"):
            pg_lora.append(p)
        elif (("norm" in ln) or ("layernorm" in ln) or ("ln" in ln)) and (("weight" in ln) or ("bias" in ln)):
            pg_norm.append(p)
        else:
            pg_other.append(p)
    groups = []
    if pg_other: groups.append({"name":"other","params": pg_other, "lr": base_lr, "weight_decay": wd_other})
    if pg_norm:  groups.append({"name":"norm", "params": pg_norm,  "lr": norm_lr, "weight_decay": 0.0})
    if pg_lora:  groups.append({"name":"lora", "params": pg_lora,  "lr": lora_lr, "weight_decay": 0.0})
    return groups

def split_param_groups_generic(model: nn.Module, base_lr: float, lora_lr: float, norm_lr: float, weight_decay: float):
    """Generic groups with same WD for 'other' and WD=0 for LoRA/Norm (safe default)."""
    pg_lora, pg_norm, pg_other = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        ln = n.lower()
        if ("lora_" in ln) or ("lora" in ln) or ln.endswith(".a") or ln.endswith(".b"):
            pg_lora.append(p)
        elif (("norm" in ln) or ("layernorm" in ln) or ("ln" in ln)) and (("weight" in ln) or ("bias" in ln)):
            pg_norm.append(p)
        else:
            pg_other.append(p)
    groups = []
    if pg_other: groups.append({"name":"other","params": pg_other, "lr": base_lr, "weight_decay": weight_decay})
    if pg_norm:  groups.append({"name":"norm", "params": pg_norm,  "lr": norm_lr, "weight_decay": 0.0})
    if pg_lora:  groups.append({"name":"lora", "params": pg_lora,  "lr": lora_lr, "weight_decay": 0.0})
    return groups

def _classify_param(name: str) -> str:
    ln = name.lower()
    if ("lora_" in ln) or ("lora" in ln) or ln.endswith(".a") or ln.endswith(".b"):
        return "lora"
    if (("norm" in ln) or ("layernorm" in ln) or ("ln" in ln)) and (("weight" in ln) or ("bias" in ln)):
        return "norm"
    return "other"

def collect_param_objects(model: nn.Module):
    groups = {"lora": [], "norm": [], "other": []}
    for n, p in model.named_parameters():
        groups[_classify_param(n)].append(p)
    return groups

# ---------------------------
# Train / Eval
# ---------------------------
@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_acc, total_ece, seen = 0.0, 0.0, 0

    for bi, (images, targets) in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches > 0 and bi >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(images)
        b = targets.size(0)
        total_acc  += (logits.argmax(1) == targets).float().sum().item()
        total_ece  += expected_calibration_error(logits, targets) * b
        seen += b

        del logits
    n = max(1, seen)
    return total_acc / n, total_ece / n

def train_one_epoch(model, loader, optimizer, device,
                    freeze_ctrl: 'FreezeController' = None,
                    monitor: GradMonitor = None, loss_fn=None,
                    clip_grad: float = 0.0,
                    lr_parity: 'DynamicLRParity' = None):
    model.train()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    t0 = time.time()
    for x, y in tqdm(loader, desc="train", leave=False):
        if freeze_ctrl:
            freeze_ctrl.apply()
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if monitor: monitor.capture_pre_step()

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        if monitor: monitor.after_backward()

        if clip_grad and clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        # dynamic LR parity uses EMA grad norms collected in monitor.after_backward()
        if lr_parity and monitor:
            lr_parity.maybe_adjust(optimizer, monitor)

        scaler.step(optimizer)
        scaler.update()

        if monitor: monitor.after_step()
    return (time.time() - t0) / max(1, len(loader))

# ---------------------------
# Head unfreeze helper
# ---------------------------
def unfreeze_classifier_head(model: nn.Module):
    for name in ("head", "fc", "classifier"):
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad = True

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["cifar10","cifar100"])
    ap.add_argument("--model", type=str, default="vit_small_patch16_224")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)

    ap.add_argument("--method", type=str, required=True,
                    choices=["norm","lora","hybrid_parallel","hybrid_layerwise","hybrid_layers"])
    ap.add_argument("--policy", type=str, default=None, help="even_lora|odd_lora|deep_lora (for hybrid_layers)")
    ap.add_argument("--attn_only", action="store_true")

    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--unfreeze", action="store_true", help="Unfreeze classifier head.")

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.0)

    ap.add_argument("--lr", type=float, default=1e-3)         # head/other LR
    ap.add_argument("--lr_lora", type=float, default=5e-4)    # lora LR
    ap.add_argument("--lr_norm", type=float, default=1e-3)    # norm LR
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--clip_grad", type=float, default=0.0)

    ap.add_argument("--log_csv", type=str, default="reports/results_week6.csv")
    ap.add_argument("--diagnostics", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    # week5 addition for periodic freeze
    ap.add_argument("--freeze_lora_steps", type=int, default=0, help="Freeze LoRA for first N batches.")
    ap.add_argument("--freeze_norm_steps", type=int, default=0, help="Freeze Norm for first N batches.")
    ap.add_argument("--alt_freeze_every", type=int, default=0, help="After warmups, alternate freezing every K batches.")
    ap.add_argument("--alt_freeze_order", type=str, default="lora-first", choices=["lora-first","norm-first"])

    # week6: dynamic LR parity knobs
    ap.add_argument("--lr_parity", type=int, default=1, help="Enable dynamic LR parity (1/0).")
    ap.add_argument("--lr_parity_period", type=int, default=20)
    ap.add_argument("--lr_parity_target", type=float, default=1.0)
    ap.add_argument("--lr_parity_min_scale", type=float, default=0.5)
    ap.add_argument("--lr_parity_max_scale", type=float, default=2.0)

    args = ap.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recipe: method => augs + scheduler style
    recipe_mode = "week3" if args.method == "hybrid_parallel" else "week4"

    # Data
    train_loader, test_loader, num_classes = make_dataloaders(
        dataset=args.dataset, img_size=args.img_size, batch_size=args.batch_size,
        workers=args.workers, recipe_mode=recipe_mode
    )

    # Model build — attach adapters BEFORE moving to device
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=num_classes)
    build_method_vit(
        model,
        method=args.method,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        policy=args.policy, attn_only=args.attn_only,
    )
    if args.unfreeze:
        unfreeze_classifier_head(model)
    model.to(device)

    freeze_ctrl = None
    if (args.freeze_lora_steps > 0) or (args.freeze_norm_steps > 0) or (args.alt_freeze_every > 0):
        freeze_ctrl = FreezeController(
            model,
            freeze_lora_steps=args.freeze_lora_steps,
            freeze_norm_steps=args.freeze_norm_steps,
            alt_every=args.alt_freeze_every,
            alt_order=args.alt_freeze_order,
        )

    # Param groups
    if args.method == "hybrid_parallel":
        param_groups = split_param_groups_week3(
            model, base_lr=args.lr, lora_lr=args.lr_lora, norm_lr=(args.lr_norm or args.lr),
            wd_other=args.weight_decay
        )
    else:
        param_groups = split_param_groups_generic(
            model, base_lr=args.lr, lora_lr=args.lr_lora, norm_lr=args.lr_norm,
            weight_decay=args.weight_decay
        )
    optimizer = optim.AdamW(param_groups)

    # Epoch-level cosine sched (kept simple and comparable to your Week 5)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Loss
    loss_fn = nn.CrossEntropyLoss()

    # Logging prep
    log_path = Path(args.log_csv); log_path.parent.mkdir(parents=True, exist_ok=True)

    # Diagnostics + LR parity
    monitor = GradMonitor(model) if args.diagnostics else None
    lr_parity = DynamicLRParity(
        target_ratio=args.lr_parity_target,
        period=args.lr_parity_period,
        min_scale=args.lr_parity_min_scale,
        max_scale=args.lr_parity_max_scale,
    ) if args.lr_parity else None

    # Train
    for epoch in range(1, args.epochs + 1):
        if monitor: monitor.zero_epoch()
        avg_step_time = train_one_epoch(
            model, train_loader, optimizer, device,
            freeze_ctrl, monitor, loss_fn, args.clip_grad, lr_parity
        )
        acc1, ece = evaluate(model, test_loader, device)
        if scheduler is not None: scheduler.step()

        diag = monitor.summarize_epoch() if monitor else {
            "grad_norm_lora_mean": float("nan"), "grad_norm_norm_mean": float("nan"),
            "update_norm_lora_mean": float("nan"), "update_norm_norm_mean": float("nan"),
            "cosine_ln_mean": float("nan"), "cosine_ln_p90": float("nan"),
        }

        peak_mem_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

        row = {
            "epoch": epoch, "dataset": args.dataset, "model": args.model,
            "method": args.method, "policy": args.policy if args.policy else "n.a.",
            "attn_only": int(args.attn_only),
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
            "lr": args.lr, "lr_lora": next((pg["lr"] for pg in optimizer.param_groups if pg.get("name")=="lora"), float('nan')),
            "lr_norm": next((pg["lr"] for pg in optimizer.param_groups if pg.get("name")=="norm"), float('nan')),
            "weight_decay": args.weight_decay, "clip_grad": args.clip_grad,
            "trainable_params": count_trainable_params(model),
            "avg_step_time_s": avg_step_time, "peak_mem_bytes": int(peak_mem_bytes),
            "acc1": acc1, "ece": ece,
            "grad_norm_lora_mean": diag["grad_norm_lora_mean"],
            "grad_norm_norm_mean": diag["grad_norm_norm_mean"],
            "update_norm_lora_mean": diag["update_norm_lora_mean"],
            "update_norm_norm_mean": diag["update_norm_norm_mean"],
            "cosine_ln_mean": diag["cosine_ln_mean"],
            "cosine_ln_p90": diag["cosine_ln_p90"],
            "freeze_lora_steps": args.freeze_lora_steps,
            "freeze_norm_steps": args.freeze_norm_steps,
            "alt_freeze_every": args.alt_freeze_every,
            "alt_freeze_order": args.alt_freeze_order,
        }

        file_exists = log_path.exists()
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(row)

        print(
            f"[Epoch {epoch}/{args.epochs}] acc1={acc1:.4f} ece={ece:.4f} step={avg_step_time:.3f}s "
            f"trainable={row['trainable_params']} "
            f"cos(mean)={diag['cosine_ln_mean']:.4f} p90={diag['cosine_ln_p90']:.4f} "
            f"gnorm L/N={diag['grad_norm_lora_mean']:.2e}/{diag['grad_norm_norm_mean']:.2e} "
            f"upd L/N={diag['update_norm_lora_mean']:.2e}/{diag['update_norm_norm_mean']:.2e} "
            f"lrs L/N={row['lr_lora']:.2e}/{row['lr_norm']:.2e}"
        )

if __name__ == "__main__":
    main()
