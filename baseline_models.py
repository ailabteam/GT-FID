r"""
Deep‑learning baselines for host‑based intrusion detection (ADFA‑LD, HADES‑HIDS, …)
===================================================================================
Implements **three** neural baselines that the GT‑FID paper compares against:

* **gru**          – Embedding → n‑layer GRU (bi‑directional) → Linear
* **cnn_lstm**     – Multi‑kernel 1‑D CNN → MaxPool → Bi‑LSTM → Linear
* **transformer**  – Token + sinusoidal positional embed → TransformerEncoder → Mean‑Pool → Linear

Each model supports binary or multi‑class classification.

Typical usage
-------------
```bash
# GRU baseline
python baseline_dl.py \
  --model gru \
  --data  data/adfa_ld.jsonl \
  --vocab data/vocab.json \
  --seq_field seq \
  --epochs 50 \
  --hidden 256 \
  --layers 2 \
  --batch_size 64 \
  --device cuda
```

```bash
# Transformer baseline with more capacity and class‑imbalance weighting
python baseline_dl.py \
  --model transformer \
  --data data/adfa_ld.jsonl \
  --vocab data/vocab.json \
  --seq_field seq \
  --hidden 512 \
  --layers 6 \
  --heads 8 \
  --dropout 0.1 \
  --epochs 80 \
  --pos_weight 5.5 \
  --device cuda
```

CLI arguments
--------------
```
--model            {gru, cnn_lstm, transformer}
--data             Path to *.jsonl (required)
--vocab            Path to vocab.json (required)
--seq_field        Key name holding the sequence (default "sequence")
--num_classes      Defaults to 2 (binary)
--hidden           Hidden/embedding dim (default 256)
--layers           #RNN / #Transformer layers (default 2)
--heads            #attention heads (transformer only)
--dropout          Dropout prob (default 0.2)
--batch_size, --epochs, --lr, --device, --num_workers, etc.
--pos_weight       Float to balance BCE loss (optional)
--stratify         Enable stratified train/val/test split (default)
--test_split       Portion for test (default 0.15) and val (same)
```

The script prints **Accuracy, Precision, Recall, Macro‑F1, AUROC** on the test set and saves the best checkpoint to `checkpoints/{model}.pt`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

# ‑‑ reproducibility
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

torch.multiprocessing.set_sharing_strategy("file_system")

# ========= Dataset ========= #
class SyscallDataset(Dataset):
    def __init__(self, records: List[dict], seq_field: str = "sequence"):
        self.samples: List[Tuple[List[int], int]] = []
        for rec in records:
            seq = rec.get(seq_field) or rec.get("seq") or rec.get("sequence")
            if not seq:
                continue
            self.samples.append((seq, rec["label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seqs_padded = pad_sequence([torch.tensor(s, dtype=torch.long) for s in seqs], batch_first=True)
    return seqs_padded, lens, torch.tensor(labels, dtype=torch.long)


# ========= Models ========= #
class GRUBaseline(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)  # concat directions
        return self.fc(self.dropout(h))


class CNNLSTMBaseline(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=0)
        ks = [3, 5, 7]
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden, k, padding=k // 2) for k in ks
        ])
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        x = self.embed(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        x = torch.mean(torch.stack(conv_outs), dim=0).transpose(1, 2)  # [B, T, H]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, layers: int, heads: int, num_classes: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos = PositionalEncoding(hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 4, dropout=dropout, activation="relu", batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        mask = (x == 0)  # pad mask True where padding
        x = self.embed(x)
        x = self.pos(x)
        x = self.enc(x, src_key_padding_mask=mask)
        x = self.norm(x)
        # mean pool excluding pads
        lengths = lengths.unsqueeze(1)
        summed = torch.sum(x * (~mask).unsqueeze(-1), dim=1)
        out = summed / lengths
        return self.fc(self.dropout(out))


# ========= training utils ========= #

def compute_metrics(y_true, y_prob, num_classes: int):
    if num_classes == 2:
        y_pred = (y_prob[:, 1] > 0.5).astype(int)
    else:
        y_pred = y_prob.argmax(1)
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if num_classes == 2:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_prob[:, 1])
        except ValueError:
            metrics["auroc"] = float("nan")
    return metrics


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for seqs, lens, labels in loader:
        seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs, lens)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    y_true, y_prob = [], []
    with torch.no_grad():
        for seqs, lens, labels in loader:
            seqs, lens = seqs.to(device), lens.to(device)
            logits = model(seqs, lens)
            loss = criterion(logits, labels.to(device))
            total_loss += loss.item() * seqs.size(0)
            if num_classes == 2:
                prob = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                prob = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.append(labels.numpy())
            y_prob.append(prob)
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    metrics = compute_metrics(y_true, y_prob, num_classes)
    return total_loss / len(loader.dataset), metrics


# ========= main ========= #

def parse_args():
    p = argparse.ArgumentParser("Deep baseline trainer/evaluator")
    p.add_argument("--data", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--seq_field", default="sequence")
    p.add_argument("--model", choices=["gru", "cnn_lstm", "transformer"], default="gru")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pos_weight", type=float, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--stratify", action="store_true")
    p.add_argument("--test_split", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # ----- load data -----
    with open(args.data) as f:
        records = [json.loads(line) for line in f if line.strip()]

    labels = np.array([r["label"] for r in records])
    train_val_idx, test_idx = train_test_split(
        np.arange(len(records)), test_size=args.test_split, stratify=labels if args.stratify else None, random_state=RNG_SEED, shuffle=True,
    )
    # split train/val 10 % of train
    val_split = args.test_split
    tv_labels = labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_split, stratify=tv_labels if args.stratify else None, random_state=RNG_SEED, shuffle=True
    )

    with open(args.vocab) as f:
        vocab = json.load(f)
    vocab_size = max(vocab.values()) + 1

    train_ds = SyscallDataset([records[i] for i in train_idx], args.seq_field)
    val_ds   = SyscallDataset([records[i] for i in val_idx],   args.seq_field)
    test_ds  = SyscallDataset([records[i] for i in test_idx],  args.seq_field)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, persistent_workers=args.num_workers>0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    # ----- build model -----
    if args.model == "gru":
        model = GRUBaseline(vocab_size, args.hidden, args.layers, args.num_classes, args.dropout)
    elif args.model == "cnn_lstm":
        model = CNNLSTMBaseline(vocab_size, args.hidden, args.num_classes, args.dropout)
    else:
        model = TransformerBaseline(vocab_size, args.hidden, args.layers, args.heads, args.num_classes, args.dropout)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.num_classes == 2:
        if args.pos_weight is not None:
            pos_w = torch.tensor([1.0, args.pos_weight], device=device)
        else:
            pos_w = None
        criterion = nn.CrossEntropyLoss(weight=pos_w)
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args.num_classes)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), ckpt_dir / f"{args.model}.pt")
        print(f"Epoch {epoch:03d}: train‑loss {tr_loss:.4f} | val‑f1 {val_metrics['f1']:.3f} best {best_f1:.3f}")

    # ----- Test -----
    model.load_state_dict(torch.load(ckpt_dir / f"{args.model}.pt", map_location=device))
    _, test_metrics = evaluate(model, test_loader, criterion, device, args.num_classes)
    print("Test: ", " ".join([f"{k} {v:.3f}" for k, v in test_metrics.items()]))

if __name__ == "__main__":
    main()

