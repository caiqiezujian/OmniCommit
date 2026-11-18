"""Split source-related tasks together, optionally subsampling the training buckets."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory

BUCKETS = ("general", "switch", "stability")


class SourceGroups:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def root(self, key):
        if key not in self.parent:
            self.parent[key] = key
            self.size[key] = 1
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size.pop(b)

    def folds(self, seed, dev_fraction):
        # Smallest source ID gives a stable split even if input row order changes.
        representatives = {}
        for source in self.parent:
            root = self.root(source)
            representatives[root] = min(source, representatives.get(root, source))
        threshold = int(dev_fraction * 2**64)
        folds = {}
        for root, canonical in representatives.items():
            digest = hashlib.blake2b(f"{seed}\0{canonical}".encode(), digest_size=8).digest()
            folds[root] = "dev" if int.from_bytes(digest, "big") < threshold else "train"
        return folds


def rows(path):
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if raw.strip():
                try:
                    task = json.loads(json.loads(raw)["asr_task"])
                    ids, bucket = task["source_ids"], task["bucket"]
                    if (not isinstance(ids, list) or not ids or
                            not all(isinstance(source, str) and source for source in ids)
                            or bucket not in BUCKETS):
                        raise ValueError("expected source_ids and a valid bucket")
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{path} line {number}: {exc}") from exc
                yield ids, bucket, raw


def quotas(size, weights):
    if len(weights) != 3 or not all(math.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("--mix requires three nonnegative finite weights with positive sum")
    exact = [size * w / sum(weights) for w in weights]
    targets = [int(x) for x in exact]
    for ix in sorted(range(3), key=lambda i: exact[i] - targets[i], reverse=True)[:size - sum(targets)]:
        targets[ix] += 1
    return dict(zip(BUCKETS, targets))


def split(path, output_dir, dev_fraction, seed, train_size=None, mix=(.4, .4, .2)):
    if not 0 < dev_fraction < 1 or (train_size is not None and train_size < 1):
        raise ValueError("dev fraction must be in (0,1) and train-size must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new split directory: {output_dir}")
    groups = SourceGroups()
    for ids, _, _ in rows(path):
        groups.root(ids[0])
        for other in ids[1:]:
            groups.join(ids[0], other)
    folds = groups.folds(seed, dev_fraction)
    counts = {"train": Counter(), "dev": Counter()}
    component_sizes = Counter()
    for ids, bucket, _ in rows(path):
        root = groups.root(ids[0])
        counts[folds[root]][bucket] += 1
        component_sizes[root] += 1
    if not counts["train"] or not counts["dev"]:
        raise ValueError(f"Empty fold across {len(folds)} source groups; split original recordings before "
                         "augmentation if splices connect the corpus, or adjust --dev-fraction/--seed for small inputs")
    available = counts["train"].copy()
    target = quotas(train_size, mix) if train_size else dict(available)
    if any(n > available[b] for b, n in target.items()):
        raise ValueError(f"Requested quotas {target} exceed available training counts {dict(available)}")
    needed = target.copy()
    rng = random.Random(seed)
    report = {"seed": seed, "target_dev_fraction": dev_fraction, "source_groups": len(folds),
              "largest_group_rows": max(component_sizes.values()), "available": counts,
              "written": {"train": target, "dev": counts["dev"]}}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as stage:
        folder = Path(stage) / "split"
        folder.mkdir()
        with (folder / "train.jsonl").open("wb") as train, (folder / "dev.jsonl").open("wb") as dev:
            for ids, bucket, raw in rows(path):
                if folds[groups.root(ids[0])] == "dev":
                    dev.write(raw.rstrip(b"\r\n") + b"\n")
                else:
                    if needed.get(bucket, 0) and rng.randrange(available[bucket]) < needed[bucket]:
                        train.write(raw.rstrip(b"\r\n") + b"\n")
                        needed[bucket] -= 1
                    available[bucket] -= 1
        if any(needed.values()) or any(available.values()):
            raise ValueError("Input changed while splitting")
        (folder / "split.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        folder.rename(output_dir)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prepared", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dev-fraction", type=float, default=.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-size", type=int, help="Exact training sample count; omit to keep all training rows")
    p.add_argument("--mix", nargs=3, type=float, default=(.4, .4, .2), metavar=("GENERAL", "SWITCH", "STABILITY"))
    a = p.parse_args()
    print(json.dumps(split(a.prepared, a.output_dir, a.dev_fraction, a.seed, a.train_size, a.mix), indent=2))


if __name__ == "__main__":
    main()
