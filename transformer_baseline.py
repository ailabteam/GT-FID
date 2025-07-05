from __future__ import annotations
import argparse
import json
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")  # mitigate "too many open files"

# ------------------------------------
# Dataset
# ------------------------------------
class SequenceGraphSample:
    def __init__(self, seq: List[int], label: int):
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.label = int(label)
        # Build edge index for compatibility, though not used
        if len(seq) > 1:
            src = torch.tensor(seq[:-1], dtype=torch.long)
            dst = torch.tensor(seq[1:], dtype=torch.long)
            self.edge_index = torch.stack([src, dst], dim=0)
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)

class SyscallDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab_path: str, seq_field: str = "sequence"):
        self.samples: List[SequenceGraphSample] = []
        with open(jsonl_path) as f:
            for ln, raw in enumerate(f, 1):
                j = json.loads(raw)
                seq = j.get(seq_field) or j.get("sequence") or j.get("seq")
                if seq is None:
                    raise KeyError(f"Missing '{seq_field}' in line {ln}")
                self.samples.append(SequenceGraphSample(seq, j["label"]))
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.num_classes = len({s.label for s in self.samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ------------------------------------
# Collate fn
# ------------------------------------
PAD_IDX = 0

def collate(batch: List[SequenceGraphSample]):
    seqs = [b.seq for b in batch]
    lens = torch.tensor([len(s) for s in seqs])
    seq_pad = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor([b.label for b in batch])
    return seq_pad, lens, None, labels  # g_batch không cần thiết, trả về None

# ------------------------------------
# Model
# ------------------------------------
class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, emb_dim: int = 128, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, emb_dim))  # max_len = 1000
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=512, dropout=0.1, batch_first=True),
            num_layers=num_layers
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, seq_pad, lens, g_batch=None):
        x = self.emb(seq_pad)  # [B, seq_len, emb_dim]
        x = x + self.pos_encoder[:, :x.size(1), :]  # Add positional encoding
        mask = (seq_pad == PAD_IDX).bool()  # [B, seq_len] padding mask
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, seq_len, emb_dim]
        # Mean pooling over sequence
        mask = (~mask).unsqueeze(-1).float()  # [B, seq_len, 1]
        x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B, emb_dim]
        return self.classifier(x)

# ------------------------------------
# Train / Eval helpers
# ------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, correct, tot = 0.0, 0, 0
    for seqs, lens, _, labels in loader:
        seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
        logits = model(seqs, lens)
        loss = criterion(logits, labels)
        tot_loss += loss.item() * len(labels)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        tot += len(labels)
    return tot_loss / tot, correct / tot

def train_epoch(model, loader, optim, criterion, device):
    model.train()
    tot_loss, correct, tot = 0.0, 0, 0
    for seqs, lens, _, labels in loader:
        seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
        optim.zero_grad()
        logits = model(seqs, lens)
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()
        tot_loss += loss.item() * len(labels)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        tot += len(labels)
    return tot_loss / tot, correct / tot

# ------------------------------------
# CLI
# ------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Transformer Baseline trainer")
    p.add_argument("--data", required=True, help="JSONL with sequences/labels")
    p.add_argument("--vocab", required=True, help="Path to vocab.json")
    p.add_argument("--seq_field", default="sequence",
                   help="Key name for sequence array in JSONL (default 'sequence')")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()

# ------------------------------------
# Main
# ------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = SyscallDataset(args.data, args.vocab, args.seq_field)
    val_split = 0.1
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val],
                                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)

    model = TransformerBaseline(vocab_size=len(ds.vocab), num_classes=ds.num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {ep:03d}: train‑loss {tr_loss:.4f} acc {tr_acc:.3f} | val‑loss {val_loss:.4f} acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "transformer_best.pt")
    print(f"Best validation accuracy: {best_acc:.3f}")

if __name__ == "__main__":
    main()
