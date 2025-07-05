r"""
Baseline models for comparison with GT‑FID on host‑based intrusion‑detection datasets.

Implements four baselines described in the paper:
1. **N‑gram + TF‑IDF → Logistic Regression**  (classical bag‑of‑ngrams)
2. **GRU**                                (sequence model)
3. **CNN‑LSTM**                           (local + temporal)
4. **Transformer Encoder**                (self‑attention)

Usage
-----
```bash
python baseline_models.py \
  --data  data/adfa_ld.jsonl \
  --vocab data/vocab.json    \
  --model tfidf_logreg       # or gru | cnn_lstm | transformer
```
Common flags: `--seq_field`, `--epochs`, `--batch_size`, `--device`, `--num_workers`, `--lr`.

Design notes
------------
* Re‑uses the same JSONL format as GT‑FID. Non‑Torch baseline (tfidf_logreg) uses scikit‑learn.
* Torch baselines share an **Embedding** layer initialised with `vocab_size` and `embed_dim` (default 128).
* All baselines report *accuracy*, *precision*, *recall*, *F1*.
* Keeps code dependency‑light: PyTorch ≥2.0, scikit‑learn ≥1.3.
"""

from __future__ import annotations

import os
import json
import random
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ---------- dataset ----------
class SyscallDataset(Dataset):
    def __init__(self, jsonl_path: str, seq_field: str = "sequence"):
        self.samples: List[Tuple[List[int], int]] = []
        with open(jsonl_path) as f:
            for line in f:
                j = json.loads(line)
                seq = j.get(seq_field) or j.get("sequence") or j.get("seq")
                if seq is None:
                    raise RuntimeError(f"Missing sequence field in sample: {j.keys()}")
                self.samples.append((seq, j["label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_val_split(dataset: Dataset, val_ratio: float = 0.2, seed: int = 42):
    n = len(dataset)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    split = int(n * (1 - val_ratio))
    return torch.utils.data.Subset(dataset, idx[:split]), torch.utils.data.Subset(dataset, idx[split:])


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    seqs = [torch.tensor(s, dtype=torch.long) for s in seqs]
    seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return seqs, lens, labels

# ---------- Baseline models ----------
class GRUBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden: int = 256, num_classes: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, seqs, lens):
        x = self.embed(seqs)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h)

class CNNLSTMBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden: int = 256, kernel: int = 3, num_classes: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel, padding=kernel // 2)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, seqs, lens):
        x = self.embed(seqs).transpose(1, 2)  # B,C,T
        x = torch.relu(self.conv(x)).transpose(1, 2)  # back to B,T,C
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h)

class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, depth: int = 2, num_classes: int = 2, max_len: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, embed_dim) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, depth)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, seqs, lens):
        b, t = seqs.shape
        x = self.embed(seqs) + self.pos[:t]
        mask = (seqs == 0)
        h = self.enc(x, src_key_padding_mask=mask)
        h = h.mean(dim=1)  # global avg pool
        return self.fc(h)

# ---------- Training utilities ----------

def run_torch_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str, epochs: int, lr: float):
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for seqs, lens, labels in train_loader:
            seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
            logits = model(seqs, lens)
            loss = crit(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
        f1 = evaluate_torch(model, val_loader, device)
        if f1 > best_f1:
            best_f1 = f1
        print(f"Epoch {epoch:03d}: val‑F1 {f1:.3f}  best {best_f1:.3f}")
    return best_f1


def evaluate_torch(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for seqs, lens, labels in loader:
            logits = model(seqs.to(device), lens.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            golds.extend(labels.tolist())
    p, r, f1, _ = precision_recall_fscore_support(golds, preds, average="macro", zero_division=0)
    return f1

# ---------- TF‑IDF + LogReg pipeline ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def run_tfidf_logreg(train_ds: SyscallDataset, val_ds: SyscallDataset, ngram: int = 3):
    def seq_to_str(seq):
        return " ".join(map(str, seq))
    X_train = [seq_to_str(s) for s, _ in train_ds]
    y_train = [y for _, y in train_ds]
    X_val   = [seq_to_str(s) for s, _ in val_ds]
    y_val   = [y for _, y in val_ds]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, ngram), min_df=1)
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xva)
    p, r, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro", zero_division=0)
    acc = accuracy_score(y_val, preds)
    print(f"TF‑IDF+LR   acc {acc:.3f}  F1 {f1:.3f}  P {p:.3f}  R {r:.3f}")
    return f1

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("Baseline trainer/evaluator")
    ap.add_argument("--data", required=True, help="Path to JSONL dataset")
    ap.add_argument("--vocab", required=True, help="Path to vocab.json (for vocab size)")
    ap.add_argument("--seq_field", default="sequence")
    ap.add_argument("--model", choices=["tfidf_logreg", "gru", "cnn_lstm", "transformer"], default="tfidf_logreg")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    ds = SyscallDataset(args.data, seq_field=args.seq_field)
    train_ds, val_ds = train_val_split(ds)

    if args.model == "tfidf_logreg":
        run_tfidf_logreg(train_ds, val_ds)
        return

    with open(args.vocab) as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    if args.model == "gru":
        model = GRUBaseline(vocab_size)
    elif args.model == "cnn_lstm":
        model = CNNLSTMBaseline(vocab_size)
    else:  # transformer
        model = TransformerBaseline(vocab_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    run_torch_model(model, train_loader, val_loader, args.device, args.epochs, args.lr)

if __name__ == "__main__":
    main()

