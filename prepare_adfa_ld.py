import os, json, glob, argparse
from collections import Counter
from tqdm import tqdm

def build_vocab(trace_paths):
    counter = Counter()
    for path in tqdm(trace_paths, desc="Scanning syscalls"):
        with open(path) as f:
            counter.update(map(str.strip, f))
    syscall2id = {s: i+1 for i, (s, _) in enumerate(counter.most_common())}
    syscall2id["<PAD>"] = 0
    return syscall2id

def trace_to_ids(path, syscall2id):
    with open(path) as f:
        return [syscall2id[x.strip()] for x in f if x.strip()]

def iter_traces(root):
    # normal
    for p in glob.glob(os.path.join(root, "Training_Data_Master/**/*.txt"), recursive=True):
        yield p, 0
    for p in glob.glob(os.path.join(root, "Validation_Data_Master/**/*.txt"), recursive=True):
        yield p, 0
    # attack
    for p in glob.glob(os.path.join(root, "Attack_Data_Master/**/*.txt"), recursive=True):
        yield p, 1

def main(args):
    paths_labels = list(iter_traces(args.input_dir))
    syscall2id = build_vocab([p for p, _ in paths_labels])
    with open(args.vocab, "w") as f: json.dump(syscall2id, f)
    with open(args.output, "w") as w:
        for p, y in tqdm(paths_labels, desc="Writing JSONL"):
            seq = trace_to_ids(p, syscall2id)
            w.write(json.dumps({"seq": seq, "label": y}) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output", default="adfa_ld.jsonl")
    ap.add_argument("--vocab",  default="vocab.json")
    main(ap.parse_args())

