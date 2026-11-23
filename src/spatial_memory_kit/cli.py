"""Inspect, replay, compare memory controls, and smoke-train on CPU."""
import argparse
import json
from pathlib import Path
import torch
from .model_io import load_model, read_manifest, resolve_inside, sha256
from .replay import replay, load_trace
from .reproducibility import seed_everything
from .smoke import train_smoke


def verify(root):
    entries = read_manifest(root, "models")["models"]
    if len(entries) != 24 or len({(e["id"], e["state"]) for e in entries}) != 24:
        raise ValueError("Expected 24 distinct model checkpoints")
    for entry in entries:
        model, _ = load_model(root, entry["id"], entry["state"])
        if sum(p.numel() for p in model.parameters()) != entry["parameters"]:
            raise ValueError("Parameter count mismatch")
    traces = read_manifest(root, "traces")["traces"]
    if len(traces) != 24:
        raise ValueError("Expected 24 recorded traces")
    checks = [replay(root, row["model_id"], row["route"]) for row in traces]
    passed = all(check["reference_within_tolerance"] for check in checks)
    return {"passed": passed, "models_verified": len(entries), "recorded_traces_replayed": len(checks), "reference_tolerance": 1e-5, "maximum_reference_absolute_error": max(max(c["trained_full_reference_absolute_errors"].values()) for c in checks), "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify_parser = sub.add_parser("verify", help="Check all checkpoint hashes and replay all 24 trained traces")
    verify_parser.add_argument("--root", type=Path, default=Path.cwd())
    replay_parser = sub.add_parser("replay", help="Evaluate one selected recorded episode")
    replay_parser.add_argument("--root", type=Path, default=Path.cwd())
    replay_parser.add_argument("--model", default="coupled_b7_s33101")
    replay_parser.add_argument("--route", choices=("a_one", "a_two"), default="a_one")
    replay_parser.add_argument("--state", choices=("trained", "initial"), default="trained")
    replay_parser.add_argument("--history", choices=("full", "reset-now", "reset8"), default="full")
    replay_parser.add_argument("--output", type=Path, help="Optional new .npz output file, never overwritten")
    smoke_parser = sub.add_parser("smoke-train", help="Run an independent short CPU learning check, not H2 evaluation")
    smoke_parser.add_argument("--root", type=Path, default=Path.cwd(), help="Accepted for a consistent interface; no release weights used")
    smoke_parser.add_argument("--output", type=Path, required=True)
    smoke_parser.add_argument("--updates", type=int, default=40)
    smoke_parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args(argv)
    seed_everything(0)
    if args.command == "verify":
        result = verify(args.root)
    elif args.command == "replay":
        result = replay(args.root, args.model, args.route, args.state, args.history, args.output)
    else:
        result = train_smoke(args.output, args.updates, args.seed)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "verify" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

