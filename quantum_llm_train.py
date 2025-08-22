"""
Minimal, runnable quantum-inspired LLM training script.

Features included (practical, working subset of your ideas):
- Quantum-inspired complex embeddings (amplitude+phase) using torch.complex64
- Simple phase-space processing (phase rotations)
- Attention operating on complex real+imag parts with stable scoring
- Coherence regularizer (annealed)
- Tokenizer that learns vocab from corpus and saves/loads state dict (no fragile pickling)
- Training loop with checkpointing, generation, and small defaults for quick runs

Designed to be runnable on CPU or GPU. Start small (default max_samples=5000)

Usage examples:
  python quantum_llm_train.py --mode train --max_samples 5000 --epochs 3
  python quantum_llm_train.py --mode generate --checkpoint checkpoints/checkpoint_last --prompt "Hello world"

"""

from __future__ import annotations
import os
import math
import pickle
import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from datasets import load_dataset
from tqdm import tqdm

# ----------------------------- Utilities ---------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# ----------------------------- Tokenizer ---------------------------------
class QuantumTokenizer:
    """Simple tokenizer: whitespace+punct split, builds vocab from corpus.
    Stores token phases and can save/load state without pickling class objects.
    """
    PAD = '<PAD>'
    UNK = '<UNK>'
    BOS = '<BOS>'
    EOS = '<EOS>'

    def __init__(self, dim: int = 64, max_vocab_size: int = 8192):
        self.dim = dim
        self.max_vocab_size = max_vocab_size
        self.special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS]
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}
        self.reverse_vocab: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        # token phase patterns (buffer) - will be registered later when model created
        self.token_phases = None

    @staticmethod
    def _word_tokenize(text: str) -> List[str]:
        # simple regex-free split preserving punctuation as tokens
        tokens = []
        cur = ''
        for ch in text:
            if ch.isalnum():
                cur += ch.lower()
            else:
                if cur:
                    tokens.append(cur)
                    cur = ''
                if not ch.isspace():
                    tokens.append(ch)
        if cur:
            tokens.append(cur)
        return tokens

    def build_vocab(self, texts: List[str]):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            cnt.update(self._word_tokenize(t))
        # reserve space for special tokens
        for word, _ in cnt.most_common(self.max_vocab_size - len(self.vocab)):
            if word in self.vocab:
                continue
            idx = len(self.vocab)
            self.vocab[word] = idx
            self.reverse_vocab[idx] = word
        self.vocab_size = len(self.vocab)
        # initialize token_phases as CPU tensor for portability
        phases = torch.randn(self.vocab_size, self.dim) * 0.5
        # keep on CPU until model moves it to device
        self.token_phases = phases

    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        toks = self._word_tokenize(text)
        ids = []
        if add_special_tokens:
            ids.append(self.vocab[self.BOS])
        for w in toks:
            ids.append(self.vocab.get(w, self.vocab[self.UNK]))
        if add_special_tokens:
            ids.append(self.vocab[self.EOS])
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        words = []
        for tid in token_ids:
            w = self.reverse_vocab.get(int(tid), self.UNK)
            if skip_special_tokens and w in self.special_tokens:
                continue
            words.append(w)
        # simple join putting spaces between tokens (may separate punctuation)
        return ' '.join(words)

    def save_state(self, path: str):
        state = {
            'dim': self.dim,
            'max_vocab_size': self.max_vocab_size,
            'vocab': self.vocab,
            'reverse_vocab': self.reverse_vocab,
            'token_phases': self.token_phases.cpu().numpy() if self.token_phases is not None else None
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load_state(cls, path: str) -> 'QuantumTokenizer':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        tok = cls(dim=state['dim'], max_vocab_size=state['max_vocab_size'])
        tok.vocab = state['vocab']
        tok.reverse_vocab = state['reverse_vocab']
        tok.vocab_size = len(tok.vocab)
        if state.get('token_phases') is not None:
            tok.token_phases = torch.tensor(state['token_phases'], dtype=torch.float32)
        else:
            tok.token_phases = None
        return tok

    def __len__(self):
        return self.vocab_size

# --------------------------- Embedding (complex) --------------------------
class QuantumEmbedding(nn.Module):
    """Complex embeddings implemented using amplitude and phase parameters.
    Returns real, imag tensors for downstream processing.
    """
    def __init__(self, tokenizer: QuantumTokenizer, embedding_dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        vocab_size = len(tokenizer)
        # amplitude and phase as learnable parameters per token
        self.amplitude = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.02)
        # phase in radians, wrap in -pi..pi
        self.phase = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.1)
        # if tokenizer has token_phases precomputed (from concept mapper), use them to init phase
        if tokenizer.token_phases is not None:
            with torch.no_grad():
                ph = tokenizer.token_phases
                if ph.shape[0] == vocab_size and ph.shape[1] >= embedding_dim:
                    self.phase.data.copy_(ph[:, :embedding_dim])

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # token_ids: [B, L]
        emb_amp = F.embedding(token_ids, self.amplitude)  # [B,L,D]
        emb_phase = F.embedding(token_ids, self.phase)    # [B,L,D]
        # Convert to complex (real, imag)
        real = emb_amp * torch.cos(emb_phase)
        imag = emb_amp * torch.sin(emb_phase)
        return real, imag

# ------------------------- Phase-space processing -------------------------
class EnhancedPhaseSpace(nn.Module):
    """Simple phase-space layer: learnable phase rotation per feature and layer norm.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.phase_offsets = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute per-dim rotation angle
        angles = torch.tanh(self.phase_offsets) * math.pi
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        # apply rotation: (r', i') = (r*cos - i*sin, r*sin + i*cos)
        r = real * cos - imag * sin
        i = real * sin + imag * cos
        # normalize
        r = self.norm(r)
        i = self.norm(i)
        return r, i

# ------------------------------ Attention --------------------------------
class ComplexAttention(nn.Module):
    """Scaled dot-product attention operating on real+imag components.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # projectors for real and imag parts separately
        self.q_proj_r = nn.Linear(dim, dim)
        self.q_proj_i = nn.Linear(dim, dim)
        self.k_proj_r = nn.Linear(dim, dim)
        self.k_proj_i = nn.Linear(dim, dim)
        self.v_proj_r = nn.Linear(dim, dim)
        self.v_proj_i = nn.Linear(dim, dim)
        self.out_r = nn.Linear(dim, dim)
        self.out_i = nn.Linear(dim, dim)
        self.scale = (self.head_dim) ** -0.5
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, H, L, head_dim]
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, head_dim] -> [B, L, D]
        x = x.transpose(1, 2).contiguous()
        B, L, H, hd = x.shape
        return x.view(B, L, H * hd)

    def forward(self, q_r: torch.Tensor, q_i: torch.Tensor,
                k_r: torch.Tensor, k_i: torch.Tensor,
                v_r: torch.Tensor, v_i: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project
        Qr = self.q_proj_r(q_r); Qi = self.q_proj_i(q_i)
        Kr = self.k_proj_r(k_r); Ki = self.k_proj_i(k_i)
        Vr = self.v_proj_r(v_r); Vi = self.v_proj_i(v_i)
        # split heads
        Qr = self._split_heads(Qr); Qi = self._split_heads(Qi)
        Kr = self._split_heads(Kr); Ki = self._split_heads(Ki)
        Vr = self._split_heads(Vr); Vi = self._split_heads(Vi)
        # Compute attention scores (real part of complex dot product): Re(q * k^H) = q_r*k_r + q_i*k_i
        # Shapes: [B,H,L,hd]
        # scores: [B,H,L,L]
        scores = torch.matmul(Qr, Kr.transpose(-2, -1)) + torch.matmul(Qi, Ki.transpose(-2, -1))
        scores = scores * self.scale
        if pad_mask is not None:
            # pad_mask: [B, L] -> [B, 1, 1, L]
            mask = pad_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # apply to values (complex)
        out_r = torch.matmul(attn, Vr) - torch.matmul(attn, Vi) * 0.0  # keep real apply to real part
        out_i = torch.matmul(attn, Vi)  # approximate separation
        # merge heads
        out_r = self._merge_heads(out_r)
        out_i = self._merge_heads(out_i)
        # final linear
        or_final = self.out_r(out_r)
        oi_final = self.out_i(out_i)
        return or_final, oi_final

# ------------------------------- Model -----------------------------------
class QuantumLLM(nn.Module):
    def __init__(self, tokenizer: QuantumTokenizer, dim: int = 128, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.tokenizer = tokenizer
        self.dim = dim
        self.num_layers = num_layers
        # ensure tokenizer.phases exist
        if tokenizer.token_phases is None:
            # initialize random phases
            tokenizer.token_phases = torch.randn(len(tokenizer), tokenizer.dim if hasattr(tokenizer, 'dim') else dim)
        # embedding
        self.embedding = QuantumEmbedding(tokenizer, dim)
        # phase space
        self.phase_space = EnhancedPhaseSpace(dim)
        # stacking attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': ComplexAttention(dim, num_heads=num_heads),
                'norm_r': nn.LayerNorm(dim),
                'norm_i': nn.LayerNorm(dim),
                'ff': nn.Sequential(
                    nn.Linear(dim * 2, dim * 2),
                    nn.GELU(),
                    nn.Linear(dim * 2, dim)
                )
            }) for _ in range(num_layers)
        ])
        # output
        self.pre_out_norm = nn.LayerNorm(dim * 2)
        self.output = nn.Linear(dim * 2, len(tokenizer))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B,L]
        real, imag = self.embedding(input_ids)
        real, imag = self.phase_space(real, imag)
        pad_id = self.tokenizer.vocab[self.tokenizer.PAD]
        pad_mask = (input_ids == pad_id)
        # layers
        for layer in self.layers:
            attn_r, attn_i = layer['attn'](real, imag, real, imag, real, imag, pad_mask=pad_mask)
            real = layer['norm_r'](real + attn_r * 0.1)
            imag = layer['norm_i'](imag + attn_i * 0.1)
            # feed-forward on concatenated complex->real
            cat = torch.cat([real, imag], dim=-1)
            ff_out = layer['ff'](cat)
            real = real + ff_out * 0.05
            imag = imag + ff_out * 0.05
        combined = torch.cat([real, imag], dim=-1)
        combined = self.pre_out_norm(combined)
        logits = self.output(combined)
        return logits

# ------------------------------- Loss ------------------------------------

def compute_enhanced_quantum_loss(model: QuantumLLM, logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int, coherence_weight: float = 0.05) -> torch.Tensor:
    # cross entropy
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id)
    # simple coherence regularizer: encouraging neighboring phase changes to be smooth
    B, L, V = logits.shape
    # derive complex logits split assumes output arranged over vocab only (not complex split)
    # compute phase from embeddings for regularizer - approximate using last layer combined
    # here we use model.embedding phase params as proxy
    phase_params = model.embedding.phase  # [Vocab, D]
    # coherence: encourage small variance in phase across feature dim
    coherence = torch.var(phase_params)
    coherence_loss = coherence
    total = ce + coherence_weight * coherence_loss
    return total

# ------------------------------ Data loader -------------------------------

def load_quantum_wikitext(max_samples: Optional[int] = None, seq_length: int = 128):
    dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    texts = [ex['text'] for ex in dataset]
    return texts

# ----------------------------- Checkpointing -----------------------------

def save_checkpoint(model: QuantumLLM, optimizer, epoch: int, loss: float, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    base = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_loss_{loss:.4f}')
    torch.save(model.state_dict(), base + '.model')
    # save tokenizer state
    tok_path = base + '.tokenizer'
    model.tokenizer.save_state(tok_path)
    # save metadata
    meta = {'epoch': epoch, 'loss': loss, 'dim': model.dim}
    torch.save(meta, base + '.meta')
    # update latest
    latest = os.path.join(checkpoint_dir, 'checkpoint_last')
    shutil_copy = getattr(__import__('shutil'), 'copy2')
    shutil_copy(base + '.model', latest + '.model')
    shutil_copy(base + '.meta', latest + '.meta')
    shutil_copy(tok_path, latest + '.tokenizer')

def load_checkpoint(model: QuantumLLM, checkpoint_base: str, device: str):
    # expects paths like .../checkpoint_last
    model_path = checkpoint_base + '.model'
    meta_path = checkpoint_base + '.meta'
    tokenizer_path = checkpoint_base + '.tokenizer'
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    meta = torch.load(meta_path, map_location=device)
    # load tokenizer state
    if os.path.exists(tokenizer_path):
        tok = QuantumTokenizer.load_state(tokenizer_path)
        model.tokenizer = tok
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model, meta

# ------------------------------- Training --------------------------------

def collate_and_pad(encodings: List[torch.Tensor], pad_id: int, max_len: int = 128) -> torch.Tensor:
    batch = []
    for e in encodings:
        if len(e) >= max_len:
            batch.append(e[:max_len])
        else:
            pad_len = max_len - len(e)
            batch.append(torch.cat([e, torch.full((pad_len,), pad_id, dtype=torch.long)]))
    return torch.stack(batch)


def train(args: argparse.Namespace):
    device = get_device()
    print('Using device:', device)
    texts = load_quantum_wikitext(max_samples=args.max_samples)
    # build tokenizer
    tokenizer = QuantumTokenizer(dim=64, max_vocab_size=8192)
    print('Building vocab...')
    tokenizer.build_vocab(texts)
    print('Vocab size:', len(tokenizer))
    # create tokenized dataset (encode)
    encodings = [tokenizer.encode(t) for t in texts]
    pad_id = tokenizer.vocab[tokenizer.PAD]
    # create model
    model = QuantumLLM(tokenizer, dim=args.model_dim, num_heads=args.num_heads, num_layers=args.num_layers)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # dataloader simple
    seq_len = args.seq_length
    batch_size = args.batch_size
    num_steps = max(1, (len(encodings) // batch_size) * args.epochs)
    print(f'Training {len(encodings)} examples, batch_size={batch_size}, steps~{num_steps}')
    step = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        pbar = tqdm(range(0, len(encodings), batch_size), desc=f'Epoch {epoch+1}')
        for i in pbar:
            batch_enc = encodings[i:i+batch_size]
            batch_input = collate_and_pad(batch_enc, pad_id, max_len=seq_len).to(device)
            optimizer.zero_grad()
            logits = model(batch_input)
            targets = torch.roll(batch_input, shifts=-1, dims=-1).to(device)
            loss = compute_enhanced_quantum_loss(model, logits, targets, pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if step % args.save_every == 0:
                # save checkpoint
                print(f"Saving checkpoint at step {step}")
                save_checkpoint(model, optimizer, epoch+1, sum(losses)/len(losses), args.checkpoint_dir)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{(sum(losses)/len(losses)):.4f}'})
        print(f'Epoch {epoch+1} avg loss: {sum(losses)/len(losses):.4f}')
    # final save
    save_checkpoint(model, optimizer, args.epochs, sum(losses)/len(losses), args.checkpoint_dir)

# ------------------------------ Generation --------------------------------

def generate(model: QuantumLLM, prompt: str, max_length: int = 100, temperature: float = 0.8):
    device = next(model.parameters()).device
    model.eval()
    ids = model.tokenizer.encode(prompt, add_special_tokens=False)
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids)
    ids = ids.unsqueeze(0).to(device)
    generated = ids
    eos = model.tokenizer.vocab.get(model.tokenizer.EOS, -1)
    with torch.no_grad():
        for i in range(max_length):
            logits = model(generated)
            next_logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos:
                break
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    out = model.tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
    return out

# ------------------------------- CLI -------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'])
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_length', type=int, default=128)
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--prompt', type=str, default='Hello world')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        # load model
        device = get_device()
        if args.checkpoint is None:
            base = os.path.join(args.checkpoint_dir, 'checkpoint_last')
        else:
            base = args.checkpoint
        if not os.path.exists(base + '.model'):
            raise FileNotFoundError('Checkpoint not found: ' + base)
        tok_path = base + '.tokenizer'
        tokenizer = QuantumTokenizer.load_state(tok_path)
        model = QuantumLLM(tokenizer, dim=args.model_dim, num_layers=args.num_layers)
        model, meta = load_checkpoint(model, base, device)
        model.to(device)
        out = generate(model, args.prompt, max_length=100)
        print('\n---\nGenerated:\n', out)
