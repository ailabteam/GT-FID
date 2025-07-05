import argparse
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# ------------------------------------
# Dataset
# ------------------------------------
class SyscallDataset:
    def __init__(self, jsonl_path: str, seq_field: str = "sequence"):
        self.sequences = []
        self.labels = []
        with open(jsonl_path) as f:
            for ln, raw in enumerate(f, 1):
                j = json.loads(raw)
                seq = j.get(seq_field) or j.get("sequence") or j.get("seq")
                if seq is None:
                    raise KeyError(f"Missing '{seq_field}' in line {ln}")
                self.sequences.append(" ".join(map(str, seq)))  # Convert to space-separated string
                self.labels.append(int(j["label"]))
        self.num_classes = len(set(self.labels))

# ------------------------------------
# CLI
# ------------------------------------
def parse_args():
    p = argparse.ArgumentParser("N-gram + TF-IDF with Logistic Regression Baseline")
    p.add_argument("--data", required=True, help="JSONL with sequences/labels")
    p.add_argument("--seq_field", default="sequence",
                   help="Key name for sequence array in JSONL (default 'sequence')")
    p.add_argument("--max_features", type=int, default=5000, help="Max number of TF-IDF features")
    return p.parse_args()

# ------------------------------------
# Main
# ------------------------------------
def main():
    args = parse_args()
    ds = SyscallDataset(args.data, args.seq_field)

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        ds.sequences, ds.labels, test_size=0.1, random_state=42
    )

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), max_features=args.max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_val_tfidf)
    val_acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {val_acc:.3f}")

    # Save model and vectorizer
    joblib.dump(model, "ngram_lr_best.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

if __name__ == "__main__":
    main()
