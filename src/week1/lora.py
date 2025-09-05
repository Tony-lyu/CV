import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """A drop-in Linear with LoRA adaptation.
    y = x W^T + scale * B(A(x))
    where A: in->r, B: r->out, scale = alpha / r.
    Base weight W is frozen.
    """
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r if r > 0 else 0.0

        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout and lora_dropout > 0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

    def forward(self, x):
        y = self.base(x)
        if self.r > 0:
            y = y + self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))
        return y

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias
