"""Replay one preselected recorded example, not the original aggregate study."""
from pathlib import Path
import numpy as np
import torch
from .model_io import load_model, read_manifest, resolve_inside, sha256


def load_trace(root, model_id, route):
    entries = read_manifest(root, "traces")["traces"]
    matches = [r for r in entries if r["model_id"] == model_id and r["route"] == route]
    if len(matches) != 1:
        raise ValueError("Expected one trace with this model and route")
    entry = matches[0]
    path = resolve_inside(root, entry["file"])
    if sha256(path) != entry["sha256"]:
        raise ValueError("Trace checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        trace = {key: archive[key].copy() for key in archive.files}
    return trace, entry


def observations(trace):
    """Only observations at t <= 48 enter the model; extra body frames are targets."""
    n = len(trace["observation_image"])
    image = torch.from_numpy(trace["observation_image"])[None, :, None].float()
    # Trace images already lie in [0, 1]. Do not divide by 255 again.
    body = torch.from_numpy(trace["body"][:n])[None].float()
    previous_action = torch.from_numpy(trace["previous_action"][:n])[None].float()
    if image.shape != (1, n, 1, 16, 16) or body.shape != (1, n, 5) or previous_action.shape != (1, n, 4):
        raise ValueError("Unexpected recorded observation shape")
    return image, body, previous_action


def infer(model, trace, history="full"):
    image, body, previous_action = observations(trace)
    if history not in ("full", "reset-now", "reset8"):
        raise ValueError("Unknown history control")
    with torch.no_grad():
        if history == "full":
            h, _ = model.beliefs(image, body, previous_action)
        elif history == "reset-now":
            h, _ = model.beliefs(image, body, previous_action, reset_each_step=True)
        else:
            encoded = model.encoder(image.reshape(-1, 1, 16, 16))[None]
            h = torch.cat([model.beliefs_from_encoded(encoded[:, max(0, t - 7):t + 1], body[:, max(0, t - 7):t + 1], previous_action[:, max(0, t - 7):t + 1])[0][:, -1:] for t in range(image.shape[1])], dim=1)
        values = {"h": h[0].cpu().numpy()}
        for horizon in (1, 5):
            actions = torch.from_numpy(trace[f"planned_action_h{horizon}"])[None].long()
            latent = model.imagine(h, actions)
            decoded = model.decode(model.rollout_decoder(latent))
            values[f"z{horizon}"] = latent[0].cpu().numpy()
            values[f"image_h{horizon}"] = decoded["image"][0, :, 0].cpu().numpy()
    return values


def replay(root, model_id="coupled_b7_s33101", route="a_one", state="trained", history="full", output=None):
    model, config = load_model(root, model_id, state)
    trace, entry = load_trace(root, model_id, route)
    values = infer(model, trace, history)
    full = values if history == "full" else infer(model, trace, "full")
    metrics = {}
    for horizon in (1, 5):
        truth = trace["truth_image" if horizon == 1 else "truth_image_h5"]
        # All 49 timepoints in this one selected episode, not branch-choice accuracy.
        metrics[f"h{horizon}_image_mse"] = float(np.mean((values[f"image_h{horizon}"] - truth) ** 2, dtype=np.float64))
        metrics[f"h{horizon}_current_image_copy_mse"] = float(np.mean((trace["observation_image"] - truth) ** 2, dtype=np.float64))
    errors = None
    if state == "trained" and history == "full":
        errors = {key: float(np.max(np.abs(values[key] - trace[reference]))) for key, reference in [("h", "h"), ("z1", "z1"), ("z5", "z5"), ("image_h1", "prediction_image"), ("image_h5", "prediction_image_h5")]}
    result = {
        "model_id": model_id, "state": state, "route": route, "history": history,
        "scope": "49 timepoints from one preselected case_index=0, query=48 recorded episode; illustration subset, not aggregate H2 or route-choice evaluation",
        "observations": len(trace["observation_image"]), "metrics": metrics,
        "maximum_hidden_difference_from_full": float(np.max(np.abs(values["h"] - full["h"]))),
        "trained_full_reference_absolute_errors": errors,
        "reference_tolerance": 1e-5 if errors is not None else None,
        "reference_within_tolerance": (max(errors.values()) <= 1e-5) if errors is not None else None,
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "runtime": {"torch": torch.__version__, "device": "cpu", "dtype": "float32", "tf32": False},
    }
    if output is not None:
        output = Path(output)
        if output.suffix.lower() != ".npz":
            raise ValueError("Replay output must have a .npz extension")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive file creation prevents both implicit extension overwrite and
        # another process winning the interval between an exists check and write.
        with output.open("xb") as handle:
            np.savez_compressed(handle, **values)
    return result

