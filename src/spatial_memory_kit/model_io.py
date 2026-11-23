"""Pickle-free model loading. Paths are relative to a downloaded release root."""
import hashlib
import json
from pathlib import Path
from safetensors.torch import load_file
from .model import PriorCorrectionBrain


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest path must remain inside the release root")
    return path


def read_manifest(root, kind):
    return json.loads((Path(root) / kind / "manifest.json").read_text(encoding="utf-8"))


def load_model(root, model_id="coupled_b7_s33101", state="trained", device="cpu"):
    if state not in ("trained", "initial"):
        raise ValueError("state must be trained or initial")
    entries = read_manifest(root, "models")["models"]
    matches = [m for m in entries if m["id"] == model_id and m["state"] == state]
    if len(matches) != 1:
        raise ValueError("Expected one model with this id and state")
    entry = matches[0]
    weights_path = resolve_inside(root, entry["weights"])
    config_path = resolve_inside(root, entry["config"])
    if sha256(weights_path) != entry["sha256"] or sha256(config_path) != entry["config_sha256"]:
        raise ValueError("Model or configuration checksum mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = PriorCorrectionBrain(**config["architecture"]).to(device)
    model.load_state_dict(load_file(str(weights_path), device=str(device)), strict=True)
    model.eval()
    return model, config

