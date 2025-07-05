from __future__ import annotations
import argparse
import json
import os
import random
from typing import List, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch_geometric.data import Data as GeoData, Batch as GeoBatch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, confusion_matrix

mp.set_sharing_strategy("file_system")

# ------------------------------------
# Dataset
# ------------------------------------
class SequenceGraphSample:
    def __init__(self, seq: List[int], label: int, max_len: int = 200):
        seq = seq[:max_len]
        # Data augmentation: drop 10% of system calls, but ensure at least one remains
        if len(seq) > 5 and random.random() < 0.5:  # Apply only if seq_len > 5 and 50% chance
            seq = [s for s in seq if random.random() > 0.1]
            seq = seq if len(seq) > 0 else [seq[0]]  # Keep at least one system call
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.label = int(label)
        if len(seq) > 1:
            src = torch.tensor(seq[:-1], dtype=torch.long)
            dst = torch.tensor(seq[1:], dtype=torch.long)
            self.edge_index = torch.stack([src, dst], dim=0)
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)

class SyscallDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab_path: str, seq_field: str = "sequence", max_len: int = 200):
        self.samples: List[SequenceGraphSample] = []
        with open(jsonl_path) as f:
            for ln, raw in enumerate(f, 1):
                j = json.loads(raw)
                seq = j.get(seq_field) or j.get("sequence") or j.get("seq")
                if seq is None:
                    raise KeyError(f"Missing '{seq_field}' in line {ln}")
                self.samples.append(SequenceGraphSample(seq, j["label"], max_len))
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.num_classes = len({s.label for s in self.samples})
        print("Class distribution:", Counter(s.label for s in self.samples))

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
    graphs = []
    for i, b in enumerate(batch):
        g = GeoData(x=b.seq.unsqueeze(1).float(), edge_index=b.edge_index)
        g.y = torch.tensor([b.label])
        graphs.append(g)
    g_batch = GeoBatch.from_data_list(graphs)
    labels = torch.tensor([b.label for b in batch])
    return seq_pad, lens, g_batch, labels

# ------------------------------------
# Model
# ------------------------------------
class GTBlock(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, lstm_hidden: int = 128, g_hidden: int = 128, fusion_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(emb_dim, lstm_hidden // 2, batch_first=True, bidirectional=True)
        self.temp_proj = nn.Linear(lstm_hidden, fusion_dim)
        self.gcn1 = GCNConv(1, g_hidden)
        self.gcn2 = GCNConv(g_hidden, g_hidden)
        self.graph_proj = nn.Linear(g_hidden, fusion_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, seq_pad: torch.Tensor, lengths: torch.Tensor, g_batch: GeoBatch):
        # Temporal branch
        x = self.emb(seq_pad)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        tx = self.dropout(self.temp_proj(h))

        # Graph branch
        gx = F.relu(self.gcn1(g_batch.x, g_batch.edge_index))
        gx = F.relu(self.gcn2(gx, g_batch.edge_index))
        gx = global_mean_pool(gx, g_batch.batch)
        gx = self.dropout(self.graph_proj(gx))

        return F.relu(gx + tx)

class GTFID(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.block = GTBlock(vocab_size)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, seq_pad, lens, g_batch):
        h = self.block(seq_pad, lens, g_batch)
        return self.classifier(h)

# ------------------------------------
# Train / Eval helpers
# ------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, correct, tot = 0.0, 0, 0
    all_preds, all_labels = [], []
    for seqs, lens, g_batch, labels in loader:
        seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
        g_batch = g_batch.to(device)
        logits = model(seqs, lens, g_batch)
        loss = criterion(logits, labels)
        tot_loss += loss.item() * len(labels)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        tot += len(labels)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = correct / tot
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    return tot_loss / tot, acc, f1, cm

def train_epoch(model, loader, optim, criterion, device):
    model.train()
    tot_loss, correct, tot = 0.0, 0, 0
    for seqs, lens, g_batch, labels in loader:
        seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
        g_batch = g_batch.to(device)
        optim.zero_grad()
        logits = model(seqs, lens, g_batch)
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
    p = argparse.ArgumentParser("Graph-Temporal Fusion (GT-FID) trainer")
    p.add_argument("--data", required=True, help="JSONL with sequences/labels")
    p.add_argument("--vocab", required=True, help="Path to vocab.json")
    p.add_argument("--seq_field", default="sequence")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_len", type=int, default=200)
    return p.parse_args()

# ------------------------------------
# Main
# ------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = SyscallDataset(args.data, args.vocab, args.seq_field, args.max_len)
    class_counts = Counter(s.label for s in ds.samples)
    total_samples = len(ds)
    class_weights = torch.tensor([(total_samples / (ds.num_classes * class_counts[i])) ** 2 for i in range(ds.num_classes)], dtype=torch.float).to(device)
    print("Class weights:", class_weights)

    val_split = 0.2
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

    model = GTFID(vocab_size=len(ds.vocab), num_classes=ds.num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)

    best_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, criterion, device)
        val_loss, val_acc, val_f1, val_cm = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {ep:03d}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}")
        print(f"Confusion matrix:\n{val_cm}")
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            torch.save(model.state_dict(), "gt_fid_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {ep}")
                break
    print(f"Best validation accuracy: {best_acc:.3f}")

if __name__ == "__main__":
    main()
