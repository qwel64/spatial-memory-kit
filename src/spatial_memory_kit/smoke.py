"""Small independent learning check; does not reproduce v33 training or H2."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from .brain import prediction_loss
from .environment import World
from .policy import ExplorationPolicy
from .model import PriorCorrectionBrain, architecture
from .reproducibility import seed_everything


def collect(layouts, episodes, episode_seed, steps=32):
    images, bodies, actions = [], [], []
    for i, layout in enumerate(layouts):
        world = World(layout)
        for j in range(episodes):
            seed = episode_seed + i * 1000 + j
            obs = world.reset(seed)
            policy = ExplorationPolicy(seed + 10000000)
            vi, bo, ac = [obs["image"]], [obs["body"]], []
            for _ in range(steps):
                action = policy.act(obs)
                obs = world.step(action)
                vi.append(obs["image"])
                bo.append(obs["body"])
                ac.append(action)
            images.append(vi)
            bodies.append(bo)
            actions.append(ac)
    # No coordinates, maps, labels, or route ids are collected or fed to training.
    return {"image": np.rint(np.asarray(images) * 255).astype(np.uint8), "body": np.asarray(bodies, np.float32), "action": np.asarray(actions, np.int64)}


def tensors(data):
    result = {k: torch.from_numpy(v) for k, v in data.items()}
    result["image"] = result["image"].float() / 255
    result["previous_action"] = torch.cat((torch.zeros(len(result["action"]), 1, 4), F.one_hot(result["action"], 4).float()), dim=1)
    return result


def evaluate_loss(model, data):
    length = data["image"].shape[1] - 5
    belief, _ = model.beliefs(data["image"][:, :length], data["body"][:, :length], data["previous_action"][:, :length])
    losses, metrics = [], {}
    for horizon in (1, 5):
        actions = torch.stack([data["action"][:, i:i + horizon] for i in range(length)], dim=1)
        prediction = model.predict(belief, actions, horizon)
        truth = data["image"][:, horizon:horizon + length]
        loss, parts = prediction_loss(prediction, truth, data["body"][:, horizon:horizon + length])
        losses.append(loss)
        metrics[f"h{horizon}_image_mse"] = float(parts["image_mse"].detach())
        metrics[f"h{horizon}_current_image_copy_mse"] = float(F.mse_loss(data["image"][:, :length], truth))
    return torch.stack(losses).mean(), metrics


def train_smoke(output, updates=40, seed=20261124):
    if updates < 1 or updates > 1000:
        raise ValueError("Smoke updates must be between 1 and 1000")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a smoke run")
    output.mkdir(parents=True)
    seed_everything(seed)
    specification = {"purpose": "Independent execution and learning smoke test; not v33 replication or H2 evidence", "seed": seed, "updates": updates, "batch_size": 4, "learning_rate": 0.0003, "steps_per_episode": 32, "train_layouts": [910001, 910002], "validation_layouts": [910101, 910102], "train_episode_seed": 930001, "validation_episode_seed": 940001, "train_episodes_per_layout": 4, "validation_episodes_per_layout": 2, "architecture": architecture("coupled_b2")}
    train_arrays = collect(specification["train_layouts"], 4, specification["train_episode_seed"])
    val_arrays = collect(specification["validation_layouts"], 2, specification["validation_episode_seed"])
    for name, arrays in (("train", train_arrays), ("validation", val_arrays)):
        np.savez_compressed(output / f"{name}.npz", **arrays)
    train_data, val_data = tensors(train_arrays), tensors(val_arrays)
    model = PriorCorrectionBrain(**specification["architecture"])
    optimizer = torch.optim.Adam(model.parameters(), lr=specification["learning_rate"])
    records = []
    model.eval()
    with torch.no_grad():
        initial_loss, initial_metrics = evaluate_loss(model, val_data)
    records.append({"update": 0, "validation_loss": float(initial_loss), **initial_metrics})
    for update in range(1, updates + 1):
        model.train()
        indices = torch.randint(len(train_data["action"]), (specification["batch_size"],))
        batch = {k: v[indices] for k, v in train_data.items()}
        loss, _ = evaluate_loss(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 10 == 0 or update == updates:
            model.eval()
            with torch.no_grad():
                validation_loss, metrics = evaluate_loss(model, val_data)
            records.append({"update": update, "training_loss": float(loss.detach()), "validation_loss": float(validation_loss), **metrics})
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(output / "smoke.safetensors"), metadata={"purpose": "independent smoke, not release v33 checkpoint"})
    report = {"completed": True, "purpose": specification["purpose"], "runtime": {"torch": torch.__version__, "device": "cpu"}, "records": records, "training_labels": ["future image", "future body"], "model_inputs": ["observed image", "observed body", "previous action"], "copy_baseline_beaten_at_final": {f"h{h}": records[-1][f"h{h}_image_mse"] < records[-1][f"h{h}_current_image_copy_mse"] for h in (1, 5)}, "no_spatial_memory_claim": True}
    (output / "config.json").write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report

