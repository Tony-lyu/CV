import torch
import torch.nn as nn
import timm
from .lora import LoRALinear

def replace_linear_with_lora(module: nn.Module, r: int, alpha: int, dropout: float, exclude_names=('head',)):
    """Recursively replace nn.Linear with LoRALinear, excluding modules whose name path contains any exclude_names.
    """
    def _should_exclude(name_path):
        return any(ex in name_path for ex in exclude_names)

    for name, child in list(module.named_children()):
        full_name = name
        if isinstance(child, nn.Linear):
            if not _should_exclude(full_name):
                lora = LoRALinear(child.in_features, child.out_features, r=r, lora_alpha=alpha, lora_dropout=dropout, bias=(child.bias is not None))
                with torch.no_grad():
                    lora.base.weight.copy_(child.weight)
                    if child.bias is not None:
                        lora.base.bias.copy_(child.bias)
                setattr(module, name, lora)
        else:
            replace_linear_with_lora(child, r, alpha, dropout, exclude_names=exclude_names)

def set_normtune(model: nn.Module, enable: bool = True):
    for n, p in model.named_parameters():
        if enable and ("norm" in n or "bn" in n or "ln" in n or "LayerNorm" in n or n.endswith("bias") and ("norm" in n)):
            pass
    # Freeze all to start
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze LayerNorm gamma/beta
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

def set_head_trainable(model: nn.Module, trainable: bool = True):
    if hasattr(model, 'head') and isinstance(model.head, nn.Module):
        for p in model.head.parameters():
            p.requires_grad = trainable

def build_vit(method: str = 'lora', pretrained: bool = True, lora_r: int = 8, lora_alpha: int = 16,
              lora_dropout: float = 0.05, head_trainable: bool = True, exclude_names=('head',),
               backbone: str = 'vit_base_patch16_224'):
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=10) 
    # Default: freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # LoRA / NormTune hooks
    if method in ('lora', 'hybrid'):
        replace_linear_with_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, exclude_names=exclude_names)

    if method in ('norm', 'hybrid'):
        set_normtune(model, enable=True)

    if head_trainable and hasattr(model, 'head'):
        for p in model.head.parameters():
                p.requires_grad = True

    return model
