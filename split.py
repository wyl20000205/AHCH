#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


DEFAULT_DATASETS = {
    "MIRFLICKR25K": 24_581,
    "NUSWIDE": 195_834,
    "MSCOCO": 123_289,
}
DEFAULT_SEEDS = [2024, 2025, 2026, 2027, 2028]


def parse_dataset_specs(values: Iterable[str]) -> Dict[str, int]:
    datasets: Dict[str, int] = {}
    for value in values:
        try:
            name, size_text = value.rsplit("=", 1)
            size = int(size_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid dataset specification {value!r}; expected NAME=SIZE."
            ) from exc

        name = name.strip()
        if not name or size <= 0:
            raise argparse.ArgumentTypeError(
                f"Invalid dataset specification {value!r}; name and size are required."
            )
        if name in datasets:
            raise argparse.ArgumentTypeError(f"Duplicate dataset name: {name}")
        datasets[name] = size
    return datasets


def dataset_rng(seed: int, dataset_name: str) -> np.random.Generator:
    dataset_id = zlib.crc32(dataset_name.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence([seed, dataset_id]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split(
    dataset_size: int,
    query_idx: np.ndarray,
    train_idx: np.ndarray,
    known_database_idx: np.ndarray,
) -> None:
    query = set(query_idx.tolist())
    train = set(train_idx.tolist())
    known = set(known_database_idx.tolist())

    if len(query) != len(query_idx):
        raise RuntimeError("Duplicate indices detected in the query set.")
    if len(train) != len(train_idx):
        raise RuntimeError("Duplicate indices detected in the training set.")
    if len(known) != len(known_database_idx):
        raise RuntimeError("Duplicate indices detected in the known database.")
    if query.intersection(known):
        raise RuntimeError("The query set overlaps the known database.")
    if not train.issubset(known):
        raise RuntimeError("The training set is not a subset of the known database.")
    if query.union(known) != set(range(dataset_size)):
        raise RuntimeError("The query set and known database do not cover the dataset.")


def generate_split(
    dataset_name: str,
    dataset_size: int,
    seed: int,
    query_size: int,
    train_size: int,
) -> Dict[str, np.ndarray]:
    if dataset_size < query_size + train_size:
        raise ValueError(
            f"{dataset_name} contains {dataset_size} samples, fewer than the "
            f"required {query_size + train_size}."
        )

    permutation = dataset_rng(seed, dataset_name).permutation(dataset_size)
    query_idx = np.sort(permutation[:query_size]).astype(np.int64)
    known_database_idx = np.sort(permutation[query_size:]).astype(np.int64)

    train_idx = np.sort(permutation[query_size : query_size + train_size]).astype(
        np.int64
    )

    validate_split(dataset_size, query_idx, train_idx, known_database_idx)
    return {
        "query_idx": query_idx,
        "train_idx": train_idx,
        "known_database_idx": known_database_idx,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=SIZE",
        help=(
            "Dataset name and number of aligned pairs. Repeat once per dataset. "
            "The manuscript dataset sizes are used when this option is omitted."
        ),
    )
    parser.add_argument("--query-size", type=int, default=5_000)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = (
        parse_dataset_specs(args.dataset) if args.dataset else DEFAULT_DATASETS.copy()
    )

    if len(datasets) < 3:
        raise ValueError("At least three datasets are required for the unknown database.")
    if args.query_size <= 0 or args.train_size <= 0:
        raise ValueError("Query and training sizes must be positive.")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Random seeds must be unique.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "protocol": {
            "query_size": args.query_size,
            "train_size": args.train_size,
            "train_is_subset_of_known_database": True,
            "query_is_disjoint_from_known_database": True,
            "unknown_database_uses_all_samples_from_other_datasets": True,
        },
        "datasets": datasets,
        "seeds": args.seeds,
        "splits": [],
    }

    records: List[Dict[str, object]] = []
    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for dataset_name, dataset_size in datasets.items():
            arrays = generate_split(
                dataset_name=dataset_name,
                dataset_size=dataset_size,
                seed=seed,
                query_size=args.query_size,
                train_size=args.train_size,
            )
            split_path = seed_dir / f"{dataset_name}.npz"
            np.savez_compressed(split_path, **arrays)

            records.append(
                {
                    "seed": seed,
                    "target_dataset": dataset_name,
                    "split_file": split_path.relative_to(output_dir).as_posix(),
                    "sha256": sha256(split_path),
                    "unknown_database_datasets": [
                        name for name in datasets if name != dataset_name
                    ],
                    "sizes": {
                        "query": len(arrays["query_idx"]),
                        "train": len(arrays["train_idx"]),
                        "known_database": len(arrays["known_database_idx"]),
                    },
                }
            )

    manifest["splits"] = records
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Created {len(records)} split files in {output_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
