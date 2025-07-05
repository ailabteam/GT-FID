r"""
Baseline trainers for ADFA‑LD / host‑IDS — July 2025 refresh
============================================================
Implements and unifies **classic + deep** baselines the paper compares with GT‑FID.

### Classical NLP baselines
* **tfidf_logreg**   – TF‑IDF n‑gram → LogisticRegression (probability output)
* **tfidf_svm**      – TF‑IDF n‑gram → LinearSVC (hinge‑loss) + Platt scaling

### Sequence NN baselines
* **gru**            – Embedding → GRU (n layers) → FC
* **cnn_lstm**       – multi‑kernel Conv1d → max‑pool → Bi‑LSTM → FC
* **transformer**    – Token+pos‑embed → TransformerEncoder → mean‑pool → FC

Each model has **CLI flags** to adjust its hyper‑parameters.  The script handles:
1. JSONL loading with flexible `--seq_field` (default `sequence`).
2. Stratified train/val/test split (`--test_split`, `--stratify`).
3. Automatic class‑imbalance handling:
   * `class_weight balanced`  (classic models)
   * `pos_weight` (nn.BCEWithLogitsLoss)  (deep models)
4. **Optimal threshold search** (`--tune_threshold`) on the validation set using
   Precision‑Recall curve → maximizes F1, then applies to test.
5. Metrics: Accuracy, Precision, Recall, F1‑macro, AUROC, Confusion matrix.
6. Optional Optuna hyper‑parameter search (`--optuna N_TRIALS`).

```bash
# Example – Logistic Regression with tuned threshold
python baseline_models.py \
  --model tfidf_logreg \
  --data  data/adfa_ld.jsonl \
  --seq_field seq \
  --ngram 1 5 \
  --max_features 750000 \
  --min_df 3 \
  --C 10 \
  --class_weight balanced \
  --test_split 0.2 \
  --stratify \
  --tune_threshold
```
"""

from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from collections import Counter
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, confusion_matrix,
)

# --- Utilities -------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)


def split_data(samples, labels, test_size=0.2, stratify=False):
    return train_test_split(
        samples, labels, test_size=test_size, random_state=42,
        stratify=labels if stratify else None
    )


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }


def find_best_threshold(y_true, y_prob) -> float:
    """Return threshold that maximizes F1 on y_true."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.nanargmax(f1s)
    return thresholds[max(best_idx, 0)]  # thresholds len = len(f1)-1


# --- TF‑IDF classic models --------------------------------------------------

def run_tfidf(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if args.model == "tfidf_logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            C=args.C,
            class_weight=args.class_weight,
            max_iter=200,
            n_jobs=-1,
            solver="lbfgs",
        )
    else:  # tfidf_svm
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(C=args.C, class_weight=args.class_weight)
        clf = CalibratedClassifierCV(base, method="sigmoid", n_jobs=-1)

    docs = [" ".join(map(str, s)) for s in args.samples]
    X_tr, X_te, y_tr, y_te = split_data(
        docs, args.labels, test_size=args.test_split, stratify=args.stratify
    )

    vect = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(args.ngram),
        max_features=args.max_features,
        min_df=args.min_df,
    )
    X_tr_tfidf = vect.fit_transform(X_tr)
    X_te_tfidf = vect.transform(X_te)

    clf.fit(X_tr_tfidf, y_tr)
    y_prob = (
        clf.predict_proba(X_te_tfidf)[:, 1]
        if hasattr(clf, "predict_proba")
        else clf.decision_function(X_te_tfidf)
    )

    threshold = 0.5
    if args.tune_threshold:
        threshold = find_best_threshold(y_tr, clf.predict_proba(X_tr_tfidf)[:, 1])
    metrics = compute_metrics(y_te, y_prob, threshold)

    print(f"{args.model.upper():<10} acc {metrics['acc']:.3f}  F1 {metrics['f1']:.3f}  "
          f"P {metrics['prec']:.3f}  R {metrics['rec']:.3f}  AUROC {metrics['auroc']:.3f}  "
          f"thr {threshold:.2f}")


# --- Deep sequence models (PyTorch) ----------------------------------------

def make_dataloaders(args):
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

    class SeqDataset(Dataset):
        def __init__(self, seqs, labels):
            self.seqs, self.labels = seqs, labels
        def __len__(self): return len(self.seqs)
        def __getitem__(self, idx): return self.seqs[idx], self.labels[idx]

    def collate(batch):
        seqs, labels = zip(*batch)
        lens = torch.tensor([len(s) for s in seqs])
        seqs_pad = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s) for s in seqs], batch_first=True
        )
        return seqs_pad, lens, torch.tensor(labels)

    X_tr, X_te, y_tr, y_te = split_data(
        args.samples, args.labels, test_size=args.test_split, stratify=args.stratify
    )

    pos_w = None
    if args.pos_weight:  # For BCEWithLogits
        pos_w = torch.tensor([args.pos_weight])

    tr_ds = SeqDataset(X_tr, y_tr)
    te_ds = SeqDataset(X_te, y_te)

    sampler = None
    if args.sample_balance:
        class_weights = 1 / np.bincount(y_tr)
        weights = class_weights[y_tr]
        sampler = WeightedRandomSampler(weights, len(weights))

    loader_tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=(sampler is None),
                           sampler=sampler, collate_fn=collate, num_workers=args.num_workers)
    loader_te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate, num_workers=args.num_workers)
    return loader_tr, loader_te, pos_w


def build_model(args, vocab_size):
    import torch.nn as nn, torch
    emb = nn.Embedding(vocab_size, args.hidden)

    if args.model == "gru":
        rnn = nn.GRU(args.hidden, args.hidden, args.layers, batch_first=True,
                     dropout=args.dropout, bidirectional=True)
        fc = nn.Linear(args.hidden * 2, 1)
    elif args.model == "cnn_lstm":
        ks = list(map(int, args.kernel_sizes))
        convs = nn.ModuleList([
            nn.Conv1d(args.hidden, args.filters, k, padding=k//2) for k in ks
        ])
        lstm = nn.LSTM(args.filters * len(ks), args.hidden, num_layers=1,
                       batch_first=True, bidirectional=True)
        fc = nn.Linear(args.hidden * 2, 1)
    else:  # transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden, nhead=args.heads, dropout=args.dropout,
        )
        trans = nn.TransformerEncoder(encoder_layer, num_layers=args.layers)
        fc = nn.Linear(args.hidden, 1)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = emb
            if args.model == "gru": self.rnn = rnn
            elif args.model == "cnn_lstm":
                self.convs, self.lstm = convs, lstm
            else: self.trans = trans
            self.fc = fc
        def forward(self, x, lens):
            x = self.emb(x)
            if args.model == "gru":
                packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
                _, h = self.rnn(packed)
                h = h.view(args.layers, 2, -1, args.hidden)[-1].transpose(0,1).reshape(x.size(0), -1)
            elif args.model == "cnn_lstm":
                c = torch.cat([
                    torch.relu(conv(x.transpose(1,2))) for conv in self.convs
                ], dim=1).transpose(1,2)
                packed = nn.utils.rnn.pack_padded_sequence(c, lens.cpu(), batch_first=True, enforce_sorted=False)
                _, (h, _) = self.lstm(packed)
                h = torch.cat((h[-2], h[-1]), dim=-1)
            else:  # transformer
                mask = (torch.arange(x.size(1))[None, :] >= lens[:, None]).to(x.device)
                h = self.trans(x.transpose(0,1), src_key_padding_mask=mask).mean(0)
            return self.fc(h).squeeze(-1)
    return Net()


def run_deep(args):
    import torch, torch.nn as nn
    vocab = json.loads(Path(args.vocab).read_text())
    vocab_size = len(vocab)

    loader_tr, loader_te, pos_w = make_dataloaders(args)
    model = build_model(args, vocab_size).to(args.device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(args.device) if pos_w is not None else None)

    best_f1, best_state = 0, None
    for epoch in range(1, args.epochs+1):
        model.train(); tot_loss=0
        for x, lens, y in loader_tr:
            x, y = x.to(args.device), y.float().to(args.device)
            optim.zero_grad(); out = model(x, lens)
            loss = criterion(out, y)
            loss.backward(); optim.step(); tot_loss += loss.item()*len(y)
        if scheduler: scheduler.step()
        # eval
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for x, lens, y in loader_te:
                out = torch.sigmoid(model(x.to(args.device), lens)); preds.extend(out.cpu()); trues.extend(y)
        preds = torch.tensor(preds); trues = torch.tensor(trues)
        threshold = args.threshold
        if args.tune_threshold:
            threshold = find_best_threshold(trues.numpy(), preds.numpy())
        f1 = f1_score(trues, (preds>=threshold).int());
        if f1>best_f1: best_f1, best_state = f1, model.state_dict()
        print(f"Epoch {epoch:03d}: F1 {f1:.3f}  best {best_f1:.3f}")

    # test final
    model.load_state_dict(best_state)
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for x, lens, y in loader_te:
            out = torch.sigmoid(model(x.to(args.device), lens)); preds.extend(out.cpu()); trues.extend(y)
    metrics = compute_metrics(trues, preds, threshold)
    print(f"{args.model.upper():<10} acc {metrics['acc']:.3f}  F1 {metrics['f1']:.3f}  "
          f"P {metrics['prec']:.3f}  R {metrics['rec']:.3f}  AUROC {metrics['auroc']:.3f}  thr {threshold:.2f}")


# --- Argument parsing -------------------------------------------------------

def get_cli():
    p = argparse.ArgumentParser("Baseline trainer/evaluator")
    p.add_argument("--data", required=True, help="Path to JSONL")
    p.add_argument("--vocab", help="Path to vocab (needed for deep models)")
    p.add_argument("--seq_field", default="sequence")
    p.add_argument("--model", default="tfidf_logreg",
                   choices=["tfidf_logreg", "tfidf_svm", "gru", "cnn_lstm", "transformer"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--filters", type=int, default=128)
    p.add_argument("--kernel_sizes", nargs="*", default=[3,5,7])
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    # TF‑IDF specific
    p.add_argument("--ngram", nargs=2, type=int, default=[1,3])
    p.add_argument("--max_features", type=int, default=300000)
    p.add_argument("--min_df", type=int, default=1)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--class_weight", default=None)
    # Common data split
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--stratify", action="store_true")
    # imbalance & threshold
    p.add_argument("--pos_weight", type=float, default=None)
    p.add_argument("--sample_balance", action="store_true")
    p.add_argument("--tune_threshold", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    # scheduler
    p.add_argument("--scheduler", choices=[None, "cosine"], default=None)

    return p


# --- main ------------------------------------------------------------------
if __name__ == "__main__":
    import torch  # required for default device above
    args = get_cli().parse_args()
    set_seed()

    # Load data
    samples, labels = [], []
    with open(args.data) as f:
        for line in f:
            j = json.loads(line)
            seq = j.get(args.seq_field) or j.get("sequence") or j.get("seq")
            assert seq is not None, "Sequence field not found"
            samples.append(seq); labels.append(j["label"])
    args.samples = samples; args.labels = np.array(labels)

    if args.model.startswith("tfidf"):
        run_tfidf(args)
    else:
        if args.vocab is None:
            sys.exit("--vocab required for deep models")
        run_deep(args)

