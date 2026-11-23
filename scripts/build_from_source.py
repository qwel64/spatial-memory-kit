"""Maintainer-only export from the original project, never needed for replay.

Run with --source-root pointing at the original research checkout. The release
does not contain the complete original training datasets or old checkpoints.
Only explicit numeric arrays and checkpoint tensor/architecture fields are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output.resolve()
    if source == output:
        raise ValueError("Source and export directories must differ")
    code = output / "src/spatial_memory_kit"
    code.mkdir(parents=True, exist_ok=True)
    provenance = []
    for src_name, dst_name in [("organism/brain.py", "brain.py"), ("organism/prior_correction.py", "model.py"), ("organism/environment.py", "environment.py"), ("organism/policy.py", "policy.py")]:
        content = (source / src_name).read_text(encoding="utf-8")
        modifications = []
        if dst_name == "model.py":
            content = content.replace("from organism.brain import Brain", "from .brain import Brain").replace("from training.train import seed_everything", "from .reproducibility import seed_everything")
            content = content.split("\ndef load_model(path, device='cpu'):")[0].rstrip() + "\n"
            modifications = ["Relative package imports", "Seed helper isolated", "Pickle loader removed; public loader uses safetensors"]
        (code / dst_name).write_text(content, encoding="utf-8")
        provenance.append({"source": src_name, "source_sha256": sha(source / src_name), "export": f"src/spatial_memory_kit/{dst_name}", "export_sha256": sha(code / dst_name), "modifications": modifications})
    models, conversions = [], []
    for arm in ("separate_b2", "coupled_b2", "separate_b7", "coupled_b7"):
        for seed in (33101, 33113, 33127):
            model_id = f"{arm}_s{seed}"
            for state in ("initial", "trained"):
                source_path = Path(f"experiments/prior_correction_v33/initial/{model_id}.pt" if state == "initial" else f"experiments/prior_correction_v33/{model_id}/block5/checkpoints/last.pt")
                checkpoint = torch.load(source / source_path, map_location="cpu", weights_only=True)
                tensors = {k: v.detach().cpu().contiguous() for k, v in checkpoint["state_dict"].items()}
                rel_weights = f"models/{model_id}/{state}.safetensors"
                rel_config = f"models/{model_id}/{state}.json"
                (output / rel_weights).parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors, str(output / rel_weights), metadata={"format": "pt", "model_id": model_id, "state": state})
                converted = load_file(str(output / rel_weights))
                exact = tensors.keys() == converted.keys() and all(torch.equal(tensors[k], converted[k]) for k in tensors)
                if not exact:
                    raise AssertionError(f"Non-exact checkpoint conversion: {model_id}/{state}")
                config = {k: checkpoint[k] for k in ("architecture", "seed", "arm", "epoch", "total_epochs", "updates", "block") if k in checkpoint}
                config.update({"release_state": state, "source_checkpoint": source_path.as_posix(), "source_checkpoint_sha256": sha(source / source_path)})
                write_json(output / rel_config, config)
                entry = {"id": model_id, "state": state, "weights": rel_weights, "config": rel_config, "sha256": sha(output / rel_weights), "config_sha256": sha(output / rel_config), "source_checkpoint_sha256": sha(source / source_path), "parameters": sum(t.numel() for t in tensors.values()), "tensor_count": len(tensors)}
                models.append(entry)
                conversions.append({"id": model_id, "state": state, "all_tensors_bit_exact": exact, "tensor_count": len(tensors)})
            original = json.loads((source / f"configs/prior_correction_{model_id}.json").read_text(encoding="utf-8"))
            # Paths describe the original study only, not data shipped in this kit.
            reference = {"scope": "Original v33 settings for reference. Full training data and schedules are not in this release.", "original_config_sha256": sha(source / f"configs/prior_correction_{model_id}.json"), "settings": original}
            write_json(output / f"configs/original_v33/{model_id}.json", reference)
    write_json(output / "models/manifest.json", {"schema_version": 1, "format": "safetensors", "models": models})
    source_manifest_path = source / "reports/project_final_video/measured_trace/manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    traces = []
    for name, item in source_manifest["traces"].items():
        src = source_manifest_path.parent / item["file"]
        arrays = {}
        with np.load(src, allow_pickle=False) as archive:
            for key in archive.files:
                value = archive[key]
                if value.dtype.kind not in "biuf":
                    raise ValueError("Only numeric trace arrays may be exported")
                arrays[key] = value
        relative = f"traces/{item['file']}"
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / relative, **arrays)
        route, model_id = name.split("/")
        traces.append({"id": name, "route": route, "model_id": model_id, "file": relative, "sha256": sha(output / relative), "source_sha256": sha(src), "source_checkpoint_sha256": item["source_checkpoint"]["sha256"], "query": item["query"], "case_index": item["case_index"], "observation_frames": int(arrays["h"].shape[0]), "reference_batch_size": item["batch_size"], "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}})
    write_json(output / "traces/manifest.json", {"schema_version": 1, "source_manifest_sha256": sha(source_manifest_path), "reference_precision": "FP32, TF32 disabled", "input_arrays": ["observation_image", "body[:observation_frames]", "previous_action"], "known_planned_actions": ["planned_action_h1", "planned_action_h5"], "display_only_arrays": ["display_cells", "display_heading", "display_targets"], "prior_nan_semantics": "prior contains NaN when prior_applied is false; prior_computed distinguishes computed-but-discarded at true episode start", "traces": traces})
    write_json(output / "models/conversion_report.json", {"source_code": provenance, "conversions": conversions, "passed": all(row["all_tensors_bit_exact"] for row in conversions)})
    print(json.dumps({"models": len(models), "traces": len(traces), "all_conversions_exact": True}))


if __name__ == "__main__":
    main()
