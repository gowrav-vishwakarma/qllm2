# quantum_llm_train.py
import argparse, os, math, time, json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ----------------------------
# System / speed knobs
# ----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.allow_tf32 = True

def device_str():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

# ----------------------------
# Dataset (char-level for simplicity & small VRAM)
# ----------------------------
class WikiTextDataset(Dataset):
    def __init__(self, split="train", seq_length=128, max_samples=None):
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        self.seq_length = seq_length
        tokens = []
        for sample in self.dataset:
            # crude char-level tokenizer (0..255)
            s = sample["text"]
            if not s: 
                continue
            tokens.extend([ord(c) % 256 for c in s])
            if max_samples and len(tokens) > max_samples * seq_length:
                break
        # pack to fixed windows
        self.data = [tokens[i:i+seq_length] for i in range(0, len(tokens)-seq_length, seq_length)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:] + [0], dtype=torch.long)
        return x, y

# ----------------------------
# Config
# ----------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 256
    dim: int = 384
    depth: int = 6
    num_heads: int = 8
    seq_length: int = 128
    subspaces: int = 3               # semantic / context / emotion
    global_tokens: int = 4           # per-head non-local tokens
    lora_rank: int = 0               # 0 disables LoRA
    lora_alpha: float = 16.0
    dropout: float = 0.0

# ----------------------------
# LoRA (optional, param-efficient)
# ----------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True, r=0, alpha=16.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r and r > 0 else 0.0
        self.base = nn.Linear(in_f, out_f, bias=bias)
        if r and r > 0:
            self.A = nn.Parameter(torch.zeros(in_f, r))
            self.B = nn.Parameter(torch.zeros(r, out_f))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        out = self.base(x)
        if self.r and self.r > 0:
            # x @ A @ B with scaling (no bias on LoRA branch)
            out = out + (x @ self.A @ self.B) * self.scaling
        return out

# ----------------------------
# Phase Utilities
# ----------------------------
PHI = (1.0 + 5 ** 0.5) / 2.0
PI  = math.pi
E   = math.e

def complex_from_amp_phase(amp, phase):
    # amp, phase: (..., D)
    return amp * torch.cos(phase), amp * torch.sin(phase)

def rotate_complex(r, i, angle):
    # elementwise rotation by 'angle'
    ca, sa = torch.cos(angle), torch.sin(angle)
    rr = r * ca - i * sa
    ii = r * sa + i * ca
    return rr, ii

# ----------------------------
# Blocks
# ----------------------------
class DynamicPhaseSubspaces(nn.Module):
    """
    Build complex representation from subspaces (semantic/context/emotion)
    with golden-ratio / pi / e driven offsets.
    """
    def __init__(self, dim, subspaces=3, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.subspaces = subspaces
        self.dropout = nn.Dropout(dropout)
        # project base embedding -> amplitude/phase per subspace
        self.proj_amp  = nn.Linear(dim, dim * subspaces)
        self.proj_phase= nn.Linear(dim, dim * subspaces)
        # gating over subspaces
        self.gate = nn.Linear(dim, subspaces)

        # learnable offsets per subspace driven by constants
        # we scale with tanh to keep angles bounded
        self.phi_scale = nn.Parameter(torch.ones(subspaces))
        self.pi_scale  = nn.Parameter(torch.zeros(subspaces))
        self.e_scale   = nn.Parameter(torch.zeros(subspaces))

    def forward(self, x):
        # x: [B,L,D]
        B, L, D = x.shape
        amp   = self.proj_amp(x).view(B, L, self.subspaces, D)
        phase = self.proj_phase(x).view(B, L, self.subspaces, D)

        # constant-driven offsets
        # angle_offset[s] = tanh(phi_s)*(pi/PHI) + tanh(pi_s)*(1/PI) + tanh(e_s)*(1/E)
        offs = torch.tanh(self.phi_scale) * (PI/PHI) + torch.tanh(self.pi_scale)*(1.0/PI) + torch.tanh(self.e_scale)*(1.0/E)
        offs = offs.view(1,1,self.subspaces,1)

        phase = torch.tanh(phase) * PI + offs   # bound phase to (-pi, pi) with offset
        r_list, i_list = [], []
        for s in range(self.subspaces):
            r, i = complex_from_amp_phase(amp[:,:,s,:], phase[:,:,s,:])
            r_list.append(r); i_list.append(i)
        r = torch.stack(r_list, dim=2)  # [B,L,S,D]
        i = torch.stack(i_list, dim=2)

        gates = F.softmax(self.gate(x), dim=-1).unsqueeze(-1)  # [B,L,S,1]
        r = (r * gates).sum(dim=2)  # [B,L,D]
        i = (i * gates).sum(dim=2)
        r = self.dropout(r); i = self.dropout(i)
        return r, i

class PhaseRotation(nn.Module):
    """Layer-wise dynamic phase rotation conditioned on sequence summary."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_angle = nn.Linear(dim, dim)

    def forward(self, r, i):
        # summarize magnitude across tokens
        mag = torch.sqrt(r.pow(2)+i.pow(2)+1e-6).mean(dim=1)  # [B,D]
        angle = torch.tanh(self.to_angle(self.norm(mag))) * (PI/PHI)
        angle = angle.unsqueeze(1)  # [B,1,D]
        return rotate_complex(r, i, angle)

class ComplexAttention(nn.Module):
    """
    Complex attention:
      scores = Re(Q * K^H) = Qr*Kr + Qi*Ki
    + global non-local tokens per head (learned K/V).
    Optional LoRA on projections.
    """
    def __init__(self, dim, num_heads, global_tokens=0, lora_rank=0, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.h = num_heads
        self.dh = dim // num_heads
        self.scale = self.dh ** -0.5
        self.dropout = nn.Dropout(dropout)

        L = lambda: LoRALinear(dim, dim, bias=False, r=lora_rank, alpha=16.0)
        self.qr = L(); self.qi = L()
        self.kr = L(); self.ki = L()
        self.vr = L(); self.vi = L()
        self.out_r = nn.Linear(dim, dim, bias=False)
        self.out_i = nn.Linear(dim, dim, bias=False)

        # global non-local tokens per head
        self.G = global_tokens
        if self.G > 0:
            # [H,G,dh] parameters (real & imag)
            self.global_kr = nn.Parameter(torch.randn(self.h, self.G, self.dh) * 0.02)
            self.global_ki = nn.Parameter(torch.randn(self.h, self.G, self.dh) * 0.02)
            self.global_vr = nn.Parameter(torch.randn(self.h, self.G, self.dh) * 0.02)
            self.global_vi = nn.Parameter(torch.randn(self.h, self.G, self.dh) * 0.02)
            self.global_mix = nn.Parameter(torch.tensor(0.5))  # how much global to mix in

    def _split(self, x):
        B,L,D = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1,2)  # [B,H,L,dh]

    def _merge(self, x):
        B,H,L,dh = x.shape
        return x.transpose(1,2).contiguous().view(B, L, H*dh)

    def forward(self, r, i, attn_mask: Optional[torch.Tensor]=None):
        # r,i: [B,L,D]
        Qr, Qi = self._split(self.qr(r)), self._split(self.qi(i))
        Kr, Ki = self._split(self.kr(r)), self._split(self.ki(i))
        Vr, Vi = self._split(self.vr(r)), self._split(self.vi(i))
        # scores real-part of complex dot
        # [B,H,L,L]
        scores = torch.matmul(Qr, Kr.transpose(-2,-1)) + torch.matmul(Qi, Ki.transpose(-2,-1))
        scores = scores * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[:,None,None,:].bool(), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # token->token
        out_r = torch.matmul(attn, Vr)
        out_i = torch.matmul(attn, Vi)

        # add global non-local (tokens -> learned globals)
        if self.G > 0:
            # Q: [B,H,L,dh], K_g: [H,G,dh] -> [B,H,L,G]
            sg = (torch.matmul(Qr, self.global_kr.transpose(-2,-1)) +
                  torch.matmul(Qi, self.global_ki.transpose(-2,-1))) * self.scale
            attn_g = F.softmax(sg, dim=-1)
            og_r = torch.matmul(attn_g, self.global_vr)  # [B,H,L,dh]
            og_i = torch.matmul(attn_g, self.global_vi)
            mix = torch.sigmoid(self.global_mix)
            out_r = out_r + mix * og_r
            out_i = out_i + mix * og_i

        out_r = self.out_r(self._merge(out_r))
        out_i = self.out_i(self._merge(out_i))
        return out_r, out_i

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim*2, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim)
        )
    def forward(self, r, i):
        x = torch.cat([r, i], dim=-1)
        y = self.net(x)
        # return same increment to r and i (phase-preserving-ish)
        return r + y, i + y

class TransformerBlockV2(nn.Module):
    def __init__(self, dim, heads, global_tokens, lora_rank, dropout=0.0):
        super().__init__()
        self.pr = nn.LayerNorm(dim)
        self.pi = nn.LayerNorm(dim)
        self.attn = ComplexAttention(dim, heads, global_tokens, lora_rank, dropout)
        self.rot  = PhaseRotation(dim)
        self.nr = nn.LayerNorm(dim); self.ni = nn.LayerNorm(dim)
        self.ff  = FeedForward(dim, mult=4, dropout=dropout)

    def forward(self, r, i, pad_mask=None):
        ar, ai = self.attn(self.pr(r), self.pi(i), attn_mask=pad_mask)
        r = r + 0.5 * ar
        i = i + 0.5 * ai
        r, i = self.rot(r, i)
        r = self.nr(r); i = self.ni(i)
        r, i = self.ff(r, i)
        return r, i

# ----------------------------
# Model
# ----------------------------
class QuantumInspiredLLM_V2(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.subspaces = DynamicPhaseSubspaces(cfg.dim, subspaces=cfg.subspaces, dropout=cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlockV2(cfg.dim, cfg.num_heads, cfg.global_tokens, cfg.lora_rank, dropout=cfg.dropout)
            for _ in range(cfg.depth)
        ])
        self.out_norm = nn.LayerNorm(cfg.dim*2)
        self.head = nn.Linear(cfg.dim*2, cfg.vocab_size, bias=False)

    def forward(self, x, pad_mask=None, return_phase=False):
        # x: [B,L]
        base = self.embed(x)              # [B,L,D]
        r, i = self.subspaces(base)       # [B,L,D] each
        for blk in self.blocks:
            r, i = blk(r, i, pad_mask=pad_mask)
        cat = torch.cat([r, i], dim=-1)
        logits = self.head(self.out_norm(cat))
        if return_phase:
            phase = torch.atan2(i, r)     # [B,L,D]
            return logits, phase
        return logits

# ----------------------------
# Loss
# ----------------------------
def phase_coherence_loss(phase, pad_mask=None):
    # encourage smooth phase across time
    # phase: [B,L,D]
    dphi = phase[:,1:,:] - phase[:,:-1,:]
    if pad_mask is not None:
        m = (~pad_mask[:,1:]).float().unsqueeze(-1)  # 1 for valid
        dphi = dphi * m
    return (dphi.pow(2)).mean()

# ----------------------------
# Training / Generation
# ----------------------------
def save_checkpoint(path, model, step, cfg, scaler_state=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "step": step,
        "config": asdict(cfg),
        "scaler": scaler_state
    }, path)

def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    cfg = ModelConfig(**ckpt["config"])
    model = QuantumInspiredLLM_V2(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt.get("step", 0)

@torch.no_grad()
def generate_text(model, prompt, max_new_tokens=100, temperature=0.8, top_k=50, device="cpu", seq_len=128):
    model.eval()
    # char-level ids
    ctx = [ord(c) % 256 for c in prompt][-seq_len:]
    x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            logits = model(x)[:, -1, :]  # [B,V]
        logits = logits / max(1e-8, temperature)
        if top_k is not None and top_k > 0:
            v, ix = torch.topk(logits, k=min(top_k, logits.shape[-1]))
            mask = logits < v[:, [-1]]
            logits[mask] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
        x = torch.cat([x, next_id], dim=1)[:, -seq_len:]
    out = "".join(chr(int(i)) for i in x[0].tolist())
    return out

def train(args):
    dev = device_str()
    device = torch.device(dev)
    print("Device:", device)

    cfg = ModelConfig(
        vocab_size=256,
        dim=args.model_dim,
        depth=args.num_layers,
        num_heads=args.num_heads,
        seq_length=args.seq_length,
        subspaces=3,
        global_tokens=args.global_tokens,
        lora_rank=args.lora_rank,
        dropout=args.dropout
    )
    assert cfg.dim % cfg.num_heads == 0, "model_dim must be divisible by num_heads"

    ds = WikiTextDataset(split="train", seq_length=args.seq_length, max_samples=args.max_samples)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(dev=="cuda"),
        persistent_workers=False,
        drop_last=True,
    )

    model = QuantumInspiredLLM_V2(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # optimizer (fused on CUDA if available)
    fused = (dev=="cuda") and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=fused)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev=="cuda"))

    ce_loss = nn.CrossEntropyLoss()
    step = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    t0 = time.time()

    model.train()
    for epoch in range(args.epochs):
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(dev=="cuda")):
                logits, phase = model(x, return_phase=True)
                loss_main = ce_loss(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                # pad mask not necessary for fixed windows; keep API for future
                loss_phase = phase_coherence_loss(phase, pad_mask=None) * args.phase_coh
                loss = loss_main + loss_phase

            scaler.scale(loss / max(1, args.accumulate_steps)).backward()

            if (step + 1) % args.accumulate_steps == 0:
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if step % 100 == 0:
                dt = time.time() - t0
                print(f"Epoch {epoch} Step {step}: Loss {loss.item():.4f} (CE {loss_main.item():.4f} + PH {loss_phase.item():.4f}) | {len(x)*max(1,args.accumulate_steps)/max(1,dt):.1f} tok/s")
                t0 = time.time()

            if step > 0 and step % args.save_every == 0:
                path = os.path.join(args.checkpoint_dir, f"model_step{step}.pt")
                save_checkpoint(path, model, step, cfg, scaler_state=scaler.state_dict())
                last = os.path.join(args.checkpoint_dir, "checkpoint_last.pt")
                save_checkpoint(last, model, step, cfg, scaler_state=scaler.state_dict())
                print(f"Saved checkpoint: {path} and checkpoint_last.pt")

            step += 1

    # final save
    last = os.path.join(args.checkpoint_dir, "checkpoint_last.pt")
    save_checkpoint(last, model, step, cfg, scaler_state=scaler.state_dict())
    print("Training done. Saved", last)

def generate(args):
    dev = device_str()
    device = torch.device(dev)
    ckpt_path = args.checkpoint or os.path.join(args.checkpoint_dir, "checkpoint_last.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model, cfg, _ = load_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    text = generate_text(
        model=model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=dev,
        seq_len=cfg.seq_length
    )
    print("\n---\nGenerated:\n", text)

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["train", "generate"])
    # training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_length", type=int, default=128)
    p.add_argument("--model_dim", type=int, default=384)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--dropout", type=float, default=0.0)
    # quantum-ish extras
    p.add_argument("--phase_coh", type=float, default=0.1, help="weight for phase coherence loss")
    p.add_argument("--global_tokens", type=int, default=4, help="per-head global non-local tokens")
    p.add_argument("--lora_rank", type=int, default=0, help="0 disables LoRA; try 8/16 on big models")
    # efficiency
    p.add_argument("--accumulate_steps", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--compile", action="store_true", help="use torch.compile if available")
    # generation
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--prompt", type=str, default="Hello world")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)

    args = p.parse_args()
    if args.mode == "train":
        train(args)
    else:
        generate(args)

if __name__ == "__main__":
    main()
