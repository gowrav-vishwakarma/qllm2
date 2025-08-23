import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ----------------------------
# Dataset
# ----------------------------
class WikiTextDataset(Dataset):
    def __init__(self, split="train", tokenizer=None, seq_length=128, max_samples=None):
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        self.seq_length = seq_length
        self.tokenizer = tokenizer or self.simple_tokenizer

        tokens = []
        for sample in self.dataset:
            tokens.extend(self.tokenizer(sample["text"]))
            if max_samples and len(tokens) > max_samples * seq_length:
                break

        self.tokens = tokens
        self.data = [tokens[i:i+seq_length] for i in range(0, len(tokens)-seq_length, seq_length)]

    def simple_tokenizer(self, text):
        return [ord(c) % 256 for c in text]  # crude char-level tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.data[idx][1:] + [0], dtype=torch.long)
        return x, y

# ----------------------------
# Model
# ----------------------------
class QuantumInspiredAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(2)  # (B, N, H, D/H)

        q = q * self.scale
        attn = torch.einsum("bnhd,bmhd->bhnm", q, k)
        attn = attn / (torch.norm(attn, dim=-1, keepdim=True) + 1e-6)  # normalization (quantum-inspired)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bmhd->bnhd", attn, v)
        out = out.reshape(B, N, D)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = QuantumInspiredAttention(dim, num_heads)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class QuantumInspiredLLM(nn.Module):
    def __init__(self, vocab_size=256, dim=384, depth=6, num_heads=8, seq_length=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.seq_length = seq_length

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.head(x)

# ----------------------------
# Training
# ----------------------------
def train(args):
    dataset = WikiTextDataset(split="train", seq_length=args.seq_length, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QuantumInspiredLLM(dim=args.model_dim, depth=args.num_layers,
                               num_heads=args.num_heads, seq_length=args.seq_length).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step}: Loss {loss.item():.4f}")

            if step % args.save_every == 0 and step > 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"model_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "checkpoint_last.pt"))
                print(f"Saved checkpoint: {ckpt_path} and checkpoint_last.pt")


            step += 1

# ----------------------------
# Text Generation
# ----------------------------
def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QuantumInspiredLLM(dim=args.model_dim, depth=args.num_layers,
                               num_heads=args.num_heads, seq_length=args.seq_length).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")

    model.eval()
    context = [ord(c) % 256 for c in args.prompt][:args.seq_length]
    x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(100):
        logits = model(x)[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        x = torch.cat([x, next_token], dim=1)[:, -args.seq_length:]

    out = "".join([chr(int(i)) for i in x[0].tolist()])
    print("Generated text:\n", out)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "generate"])
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for generation")
    parser.add_argument("--prompt", type=str, default="Hello world", help="Prompt for text generation")

    # Training args
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--seq_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--model_dim", type=int, default=384, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        generate(args)
