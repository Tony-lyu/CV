from typing import Dict, List, Optional
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias_param = None
        if base.bias is not None:
            self.bias_param = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        self.r = int(r)
        self.scaling = (lora_alpha / r) if r > 0 else 0.0
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.empty(self.in_features, self.r))
            self.lora_B = nn.Parameter(torch.empty(self.r, self.out_features))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.matmul(self.weight.t())
        if self.bias_param is not None:
            out = out + self.bias_param
        if self.r > 0:
            x_d = self.dropout(x)
            lora_update = x_d.matmul(self.lora_A).matmul(self.lora_B)
            out = out + self.scaling * lora_update
        return out

def _replace_linear_with_lora(module: nn.Module, target_modules: Optional[List[str]] = None,
                              r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0):
    for name, child in list(module.named_children()):
        should_wrap = isinstance(child, nn.Linear)
        if target_modules is not None:
            should_wrap = should_wrap and any(t in name for t in target_modules)
        if should_wrap:
            if isinstance(child, LoRALinear):
                continue
            setattr(module, name, LoRALinear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
        else:
            _replace_linear_with_lora(child, target_modules, r, lora_alpha, lora_dropout)

def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def enable_norm_affine(model: nn.Module, require_grad: bool = True):
    for m in model.modules():
        if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            if m.weight is not None:
                m.weight.requires_grad = require_grad
            if m.bias is not None:
                m.bias.requires_grad = require_grad

def _get_vit_blocks(vit: nn.Module) -> List[nn.Module]:
    if hasattr(vit, "blocks"):
        blocks = getattr(vit, "blocks")
        if isinstance(blocks, (nn.ModuleList, nn.Sequential, list, tuple)):
            return list(blocks)
    if hasattr(vit, "stages"):
        collected: List[nn.Module] = []
        stages = getattr(vit, "stages")
        for s in (stages if isinstance(stages, (list, tuple, nn.ModuleList)) else [stages]):
            if hasattr(s, "blocks"):
                blk = getattr(s, "blocks")
                if isinstance(blk, (nn.ModuleList, nn.Sequential, list, tuple)):
                    collected.extend(list(blk))
        if collected:
            return collected
    candidates = []
    for m in vit.modules():
        if hasattr(m, "attn") and hasattr(m, "mlp"):
            candidates.append(m)
    if candidates:
        return candidates
    raise ValueError("Could not locate ViT blocks (.blocks/.stages or attn+mlp).")

def select_layers(num_blocks: int, policy: str) -> Dict[int, str]:
    sel: Dict[int, str] = {}
    if policy == "even_lora":
        for i in range(num_blocks): sel[i] = "lora" if i % 2 == 0 else "norm"
    elif policy == "odd_lora":
        for i in range(num_blocks): sel[i] = "lora" if i % 2 == 1 else "norm"
    elif policy == "deep_lora":
        cutoff = int(2 * num_blocks / 3)
        for i in range(num_blocks): sel[i] = "lora" if i >= cutoff else "norm"
    else:
        for i in range(num_blocks): sel[i] = "none"
    return sel

def attach_lora_vit_block(block: nn.Module, r: int = 8, lora_alpha: float = 16.0,
                          lora_dropout: float = 0.0, attn_only: bool = False):
    if hasattr(block, "attn"):
        _replace_linear_with_lora(block.attn, None, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    if not attn_only and hasattr(block, "mlp"):
        _replace_linear_with_lora(block.mlp, None, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

def attach_norm_vit_block(block: nn.Module):
    for m in block.modules():
        if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            if m.weight is not None: m.weight.requires_grad = True
            if m.bias is not None:  m.bias.requires_grad = True

def build_method_vit(vit: nn.Module, method: str, lora_r: int = 8, lora_alpha: float = 16.0,
                     lora_dropout: float = 0.0, policy: Optional[str] = None, attn_only: bool = False):
    if method == "hybrid_layers":
        method = "hybrid_layerwise"

    blocks = _get_vit_blocks(vit)
    freeze_all(vit)

    if method == "norm":
        enable_norm_affine(vit, True)
    elif method == "lora":
        for blk in blocks:
            attach_lora_vit_block(blk, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, attn_only=attn_only)
    elif method == "hybrid_parallel":
        enable_norm_affine(vit, True)
        for blk in blocks:
            attach_lora_vit_block(blk, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, attn_only=attn_only)
    elif method == "hybrid_layerwise":
        if policy is None:
            raise ValueError("hybrid_layerwise requires --policy (even_lora|odd_lora|deep_lora).")
        mapping = select_layers(len(blocks), policy)
        for i, blk in enumerate(blocks):
            tag = mapping[i]
            if tag == "lora":
                attach_lora_vit_block(blk, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, attn_only=attn_only)
            elif tag == "norm":
                attach_norm_vit_block(blk)
    else:
        raise ValueError(f"Unknown method: {method}")

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
