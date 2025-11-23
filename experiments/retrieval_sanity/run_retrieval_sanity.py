#!/usr/bin/env python3
"""Minimal retrieval sanity loop comparing linear vs PMFlow embeddings.

This script trains two lightweight models on a synthetic symbol dataset:
    1. A baseline linear embedding network.
    2. A PMFlow-enhanced embedding network using the Pushing-Medium library.

After supervised training, both models populate a vector store (in-memory) and
answer nearest-neighbour queries. The script reports classification accuracy and
retrieval top-1 accuracy for each model so the impact of PMFlow can be observed
without wiring a full relational or planner stack.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Link the Pushing-Medium library without requiring installation.
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[1]
PMFLOW_ROOT = WORKSPACE_ROOT / "Pushing-Medium" / "programs" / "demos" / "machine_learning" / "nn_lib_v2"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(PMFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(PMFLOW_ROOT))

try:
    from pmflow_bnn.pmflow import ParallelPMField  # noqa: E402
except ImportError:
    from pmflow_bnn.pmflow import PMField as ParallelPMField  # noqa: E402

from experiments.retrieval_sanity.config_utils import load_labeled_overrides
from experiments.retrieval_sanity.provenance import collect_provenance

# ---------------------------------------------------------------------------
# Data generation utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DatasetSplit:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor


# ---------------------------------------------------------------------------
# Vector store abstraction
# ---------------------------------------------------------------------------


class VectorStore:
    """Simple in-memory vector store with cosine or Euclidean similarity."""

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unsupported metric '{metric}'.")
        self.metric = metric
        self._embeddings: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None

    def clear(self) -> None:
        self._embeddings = None
        self._labels = None

    def add(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        embeddings = embeddings.detach()
        labels = labels.detach()
        if self._embeddings is None or self._labels is None:
            self._embeddings = embeddings
            self._labels = labels
        else:
            existing_emb = cast(torch.Tensor, self._embeddings)
            existing_lbl = cast(torch.Tensor, self._labels)
            self._embeddings = torch.cat([existing_emb, embeddings], dim=0)
            self._labels = torch.cat([existing_lbl, labels], dim=0)

    def search(self, queries: torch.Tensor, topk: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._embeddings is None or self._labels is None:
            raise RuntimeError("VectorStore is empty.")

        embeddings = self._embeddings.to(queries.device)

        if self.metric == "cosine":
            support = F.normalize(embeddings, p=2, dim=1)
            q_norm = F.normalize(queries, p=2, dim=1)
            scores = q_norm @ support.T
            values, indices = scores.topk(topk, dim=1)
        else:  # euclidean
            distances = torch.cdist(queries, embeddings)
            neg_dist = -distances
            values, indices = neg_dist.topk(topk, dim=1)

        labels = self._labels.to(queries.device)
        top_labels = labels[indices]
        return values, top_labels


def make_symbol_dataset(
    num_classes: int = 3,
    points_per_class: int = 90,
    support_ratio: float = 0.6,
    cluster_std: float = 0.55,
    seed: int = 42,
) -> DatasetSplit:
    """Generate a toy dataset of clustered points representing symbolic items."""
    set_seed(seed)
    centers = []
    angle_step = 2 * math.pi / num_classes
    radius = 3.0
    for i in range(num_classes):
        angle = i * angle_step
        centers.append((radius * math.cos(angle), radius * math.sin(angle)))

    features = []
    labels = []
    for class_idx, (cx, cy) in enumerate(centers):
        points = np.random.randn(points_per_class, 2) * cluster_std
        points += np.array([cx, cy])
        features.append(points)
        labels.extend([class_idx] * points_per_class)

    X = torch.tensor(np.vstack(features), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Normalise data for stable training
    X = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-6)

    support_x_list = []
    support_y_list = []
    query_x_list = []
    query_y_list = []

    for class_idx in range(num_classes):
        class_mask = y == class_idx
        class_indices = class_mask.nonzero(as_tuple=False).squeeze(1)
        perm = class_indices[torch.randperm(class_indices.shape[0])]
        split_point = int(len(perm) * support_ratio)
        support_idx = perm[:split_point]
        query_idx = perm[split_point:]
        support_x_list.append(X[support_idx])
        support_y_list.append(y[support_idx])
        query_x_list.append(X[query_idx])
        query_y_list.append(y[query_idx])

    return DatasetSplit(
        support_x=torch.vstack(support_x_list),
        support_y=torch.hstack(support_y_list),
        query_x=torch.vstack(query_x_list),
        query_y=torch.hstack(query_y_list),
    )


def load_relational_export(
    path: Path,
    support_ratio: float = 0.6,
    seed: int = 42,
) -> DatasetSplit:
    """Load a relational export stored as JSON and split into support/query sets."""

    with path.open() as f:
        data = json.load(f)

    entries = data.get("entries", [])
    if not entries:
        raise ValueError(f"No entries found in relational export {path}.")

    class_names = sorted({entry["class"] for entry in entries})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    features = torch.tensor([entry["coords"] for entry in entries], dtype=torch.float32)
    labels = torch.tensor([class_to_idx[entry["class"]] for entry in entries], dtype=torch.long)

    # Normalise features across the entire dataset
    features = (features - features.mean(dim=0, keepdim=True)) / (features.std(dim=0, keepdim=True) + 1e-6)

    set_seed(seed)

    support_x_list: List[torch.Tensor] = []
    support_y_list: List[torch.Tensor] = []
    query_x_list: List[torch.Tensor] = []
    query_y_list: List[torch.Tensor] = []

    for class_name, class_idx in class_to_idx.items():
        class_mask = labels == class_idx
        class_indices = class_mask.nonzero(as_tuple=False).squeeze(1)
        perm = class_indices[torch.randperm(class_indices.shape[0])]
        split_point = max(1, int(len(perm) * support_ratio))
        support_idx = perm[:split_point]
        query_idx = perm[split_point:] if split_point < len(perm) else perm[split_point - 1 :]

        support_x_list.append(features[support_idx])
        support_y_list.append(labels[support_idx])
        if len(query_idx) > 0:
            query_x_list.append(features[query_idx])
            query_y_list.append(labels[query_idx])

    if not query_x_list:
        raise ValueError("Relational dataset split produced an empty query set. Increase entries or adjust support_ratio.")

    return DatasetSplit(
        support_x=torch.vstack(support_x_list),
        support_y=torch.hstack(support_y_list),
        query_x=torch.vstack(query_x_list),
        query_y=torch.hstack(query_y_list),
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LinearEmbeddingNet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.embed(x)
        logits = self.head(z)
        return logits, z


class PMFlowEmbeddingNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        n_centers: int = 24,
        pm_steps: int = 4,
        dt: float = 0.12,
        beta: float = 1.1,
        clamp: float = 3.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Tanh(),
        )
        # Try ParallelPMField args first, fall back to PMField args
        try:
            self.pm = ParallelPMField(
                d_latent=latent_dim,
                n_centers=n_centers,
                steps=pm_steps,
                dt=dt,
                beta=beta,
                clamp=clamp,
                temporal_parallel=False,
            )
        except TypeError:
            # Fallback to PMField without temporal_parallel
            self.pm = ParallelPMField(
                d_latent=latent_dim,
                n_centers=n_centers,
                steps=pm_steps,
            )
        self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z0 = self.embed(x)
        z = self.pm(z0)
        logits = self.head(z)
        return logits, z


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------


def train_classifier(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int = 600,
    lr: float = 2e-3,
    l2: float = 1e-4,
) -> Dict[str, Any]:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []
    last_loss = float("nan")
    milestone = max(1, epochs // 5)

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        if (epoch + 1) % milestone == 0 or epoch == 0:
            acc = accuracy(model, train_x, train_y)
            history.append({"epoch": float(epoch + 1), "loss": float(loss.item()), "acc": float(acc)})

    return {"final_loss": last_loss, "history": history}


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.data_mode == "synthetic":
        split = make_symbol_dataset(
            num_classes=args.num_classes,
            points_per_class=args.points_per_class,
            support_ratio=args.support_ratio,
            cluster_std=args.cluster_std,
            seed=args.seed,
        )
    else:
        relational_path = Path(args.relational_path)
        if not relational_path.exists():
            raise FileNotFoundError(f"Relational export not found: {relational_path}")
        split = load_relational_export(
            relational_path,
            support_ratio=args.support_ratio,
            seed=args.seed,
        )

    train_x = split.support_x.to(device)
    train_y = split.support_y.to(device)
    query_x = split.query_x.to(device)
    query_y = split.query_y.to(device)

    latent_dim = args.latent_dim
    input_dim = train_x.shape[1]
    num_classes = int(max(train_y.max(), query_y.max()).item() + 1)

    baseline = LinearEmbeddingNet(input_dim=input_dim, latent_dim=latent_dim, num_classes=num_classes).to(device)
    pm_model = PMFlowEmbeddingNet(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_classes=num_classes,
        n_centers=args.n_centers,
        pm_steps=args.pm_steps,
        dt=args.pm_dt,
        beta=args.pm_beta,
        clamp=args.pm_clamp,
    ).to(device)

    baseline_stats = train_classifier(
        baseline, train_x, train_y, epochs=args.epochs, lr=args.lr, l2=args.weight_decay
    )
    pm_stats = train_classifier(
        pm_model, train_x, train_y, epochs=args.epochs, lr=args.lr, l2=args.weight_decay
    )

    baseline.train(False)
    pm_model.train(False)

    scenario_base = getattr(args, "sqlite_scenario", None) or getattr(args, "config_label", None) or args.data_mode
    use_sqlite = args.vector_store == "sqlite"

    baseline_store: Any
    pm_store: Any
    baseline_scenario: Optional[str]
    pm_scenario: Optional[str]

    if use_sqlite:
        from experiments.retrieval_sanity.storage.sqlite_store import SQLiteVectorStore

        sqlite_path = Path(args.sqlite_path)
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_store = SQLiteVectorStore(sqlite_path, metric=args.metric)
        pm_store = SQLiteVectorStore(sqlite_path, metric=args.metric)
        baseline_scenario = f"{scenario_base}:baseline" if scenario_base else "baseline"
        pm_scenario = f"{scenario_base}:pmflow" if scenario_base else "pmflow"
        baseline_store.clear(scenario=baseline_scenario)
        pm_store.clear(scenario=pm_scenario)
    else:
        baseline_store = VectorStore(metric=args.metric)
        pm_store = VectorStore(metric=args.metric)
        baseline_scenario = None
        pm_scenario = None

    with torch.no_grad():
        _, baseline_support_embed = baseline(train_x)
        _, baseline_query_embed = baseline(query_x)
        _, pm_support_embed = pm_model(train_x)
        _, pm_query_embed = pm_model(query_x)

    if use_sqlite:
        baseline_store.add(baseline_support_embed, train_y, scenario=baseline_scenario)
        pm_store.add(pm_support_embed, train_y, scenario=pm_scenario)
    else:
        baseline_store.add(baseline_support_embed, train_y)
        pm_store.add(pm_support_embed, train_y)

    baseline_class_acc = accuracy(baseline, query_x, query_y)
    pm_class_acc = accuracy(pm_model, query_x, query_y)

    if use_sqlite:
        _, baseline_labels = baseline_store.search(baseline_query_embed, topk=1, scenario=baseline_scenario)
        _, pm_labels = pm_store.search(pm_query_embed, topk=1, scenario=pm_scenario)
    else:
        _, baseline_labels = baseline_store.search(baseline_query_embed, topk=1)
        _, pm_labels = pm_store.search(pm_query_embed, topk=1)

    baseline_retrieval = (baseline_labels.squeeze(1) == query_y).float().mean().item()
    pm_retrieval = (pm_labels.squeeze(1) == query_y).float().mean().item()

    results = {
        "baseline_class_accuracy": baseline_class_acc,
        "baseline_retrieval_accuracy": baseline_retrieval,
        "pmflow_class_accuracy": pm_class_acc,
        "pmflow_retrieval_accuracy": pm_retrieval,
        "support_samples": int(train_x.shape[0]),
        "query_samples": int(query_x.shape[0]),
        "latent_dim": latent_dim,
        "input_dim": int(input_dim),
        "n_centers": args.n_centers,
        "pm_steps": args.pm_steps,
        "data_mode": args.data_mode,
        "num_classes": num_classes,
        "seed": args.seed,
        "device": str(device),
        "provenance": collect_provenance(
            runner="cli",
            device=str(device),
            extra={
                "script": str(Path(__file__).resolve()),
                "args": vars(args),
            },
        ),
        "vector_store": args.vector_store,
    }

    if use_sqlite:
        sqlite_path = Path(args.sqlite_path)
        results["sqlite_path"] = str(sqlite_path.resolve())
        if scenario_base:
            results["sqlite_scenario"] = scenario_base

    if getattr(args, "config_label", None):
        results["config_label"] = args.config_label
    if getattr(args, "config_path", None):
        results["config_path"] = args.config_path
    config_overrides = getattr(args, "config_overrides", {})
    if config_overrides:
        results["config_overrides"] = config_overrides

    if args.verbose:
        print("\n=== Training snapshots ===")
        print("Baseline milestones:")
        for snapshot in baseline_stats["history"]:
            print(snapshot)
        print("PMFlow milestones:")
        for snapshot in pm_stats["history"]:
            print(snapshot)

    print("\n=== Retrieval sanity results ===")
    print(json.dumps(results, indent=2))
    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = _build_parser()
    argv = list(argv) if argv is not None else sys.argv[1:]

    preliminary, _ = parser.parse_known_args(argv)

    config_label: Optional[str] = None
    config_path: Optional[Path] = None
    config_overrides: Dict[str, str] = {}

    if getattr(preliminary, "config", None):
        config_path = Path(preliminary.config)
        config_label, raw_overrides = load_labeled_overrides(config_path)
        config_overrides = raw_overrides
        user_keys = _extract_cli_keys(argv)
        defaults: Dict[str, Any] = {}

        for action in parser._actions:
            if not action.option_strings or action.dest in {"help", "config"}:
                continue
            option_key = action.dest.replace("_", "-")
            if option_key not in raw_overrides or option_key in user_keys:
                continue
            defaults[action.dest] = _coerce_action_value(action, raw_overrides[option_key])

        if defaults:
            parser.set_defaults(**defaults)

    args = parser.parse_args(argv)

    if config_label:
        setattr(args, "config_label", config_label)
    if config_path:
        setattr(args, "config_path", str(config_path.resolve()))
    if config_overrides:
        setattr(args, "config_overrides", config_overrides)
    else:
        setattr(args, "config_overrides", {})

    return args


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PMFlow retrieval sanity experiment.")
    parser.add_argument("--config", type=str, help="Optional JSON file containing override parameters.")
    parser.add_argument(
        "--data-mode",
        choices=["synthetic", "relational"],
        default="synthetic",
        help="Choose between synthetic clusters or relational export dataset.",
    )
    parser.add_argument("--num-classes", type=int, default=3, help="Number of synthetic symbol classes.")
    parser.add_argument(
        "--points-per-class",
        type=int,
        default=90,
        help="Total samples per class before splitting into support/query.",
    )
    parser.add_argument("--support-ratio", type=float, default=0.6, help="Fraction of each class used for support store.")
    parser.add_argument("--cluster-std", type=float, default=0.55, help="Standard deviation of each Gaussian cluster.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent embedding dimensionality.")
    parser.add_argument("--n-centers", type=int, default=24, help="Number of PMFlow gravitational centers.")
    parser.add_argument("--pm-steps", type=int, default=4, help="PMFlow temporal evolution steps.")
    parser.add_argument("--pm-dt", type=float, default=0.12, help="PMFlow time step size.")
    parser.add_argument("--pm-beta", type=float, default=1.1, help="PMFlow coupling strength.")
    parser.add_argument("--pm-clamp", type=float, default=3.0, help="Clamp range for PMFlow latents.")
    parser.add_argument("--epochs", type=int, default=600, help="Training epochs for both models.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate for Adam optimizer.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--relational-path",
        type=str,
        default=str(THIS_DIR / "relational_export.json"),
        help="Path to relational export JSON used when --data-mode=relational.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Similarity metric for the vector store.",
    )
    parser.add_argument(
        "--vector-store",
        choices=["memory", "sqlite"],
        default="memory",
        help="Backend to persist support embeddings before retrieval.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=str,
        default=str(Path.cwd() / "retrieval_vectors.db"),
        help="Database file used when --vector-store=sqlite.",
    )
    parser.add_argument(
        "--sqlite-scenario",
        type=str,
        help="Optional scenario identifier to namespace SQLite entries.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate training milestones.")
    return parser


def _extract_cli_keys(argv: List[str]) -> set[str]:
    keys: set[str] = set()
    for token in argv:
        if not token.startswith("--") or token == "--":
            continue
        stripped = token[2:]
        if "=" in stripped:
            key = stripped.split("=", 1)[0]
        else:
            key = stripped
        keys.add(key)
    return keys


def _coerce_action_value(action: argparse.Action, value: str) -> Any:
    converter: Optional[Callable[[str], Any]] = getattr(action, "type", None)
    if callable(converter):
        return converter(value)
    if isinstance(action.default, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    return value


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    run_experiment(args)
