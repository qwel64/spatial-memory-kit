---
license: mit
language:
  - en
  - ja
tags:
  - world-model
  - spatial-memory
  - recurrent-neural-network
  - pytorch
  - safetensors
  - reproducibility
  - negative-results
---

# Spatial Memory Kit

A small research kit for testing spatial memory in recurrent world models.

**A model can use its history without having a spatial representation that transfers to a different route.** This release makes that distinction inspectable: 12 trained models, their 12 matched initial models, and 24 recorded trajectories with observations, predictions, and recurrent states. Each model has **276,501 parameters** and a 128-dimensional recurrent state.

The original comparison study found improved sensory prediction and limited memory use. It did **not** establish stable spatial understanding across changes in route, direction, and distance, or an integrated artificial organism. Negative results are part of the release.

This is exploratory personal research developed with AI agents for implementation, analysis, auditing, and documentation. Agent-assisted checks are not external peer review or independent laboratory replication.

[日本語の説明](README.ja.md) · [Model card](MODEL_CARD.md) · [Methods and interpretation](docs/methods.md) · [Results and limitations](docs/results.md)

[Source and kit on GitHub](https://github.com/qwel64/spatial-memory-kit) · [Models and kit on Hugging Face](https://huggingface.co/nao1357/spatial-memory-kit-v0.1) · [Recorded trace explorer](https://huggingface.co/spaces/nao1357/spatial-memory-kit)

## What is included

| Item | Scope |
|---|---|
| Models | 4 conditions × 3 paired seeds, trained and initial: 24 FP32 safetensors files |
| Architecture | 16×16 grayscale image encoder; 5 body values; previous action; 128-state GRU; action-only transition; image/body decoder |
| Recorded trajectories | 12 trained models × 2 routes; 49 observed timesteps per trace; predictions at horizons 1 and 5 |
| Inspection | State, prior, and imagined states; observations and corresponding future targets; explicit prior-computation/adoption masks |
| Reproduction | Tensor-file verification; replay with trained or initial weights; a small training smoke run |
| Documentation | Scientific scope, limits, initial-model controls, serialization provenance, and MIT license |

The four conditions are `separate_b2`, `coupled_b2`, `separate_b7`, and `coupled_b7`; paired seeds are `33101`, `33113`, and `33127`. These are **three paired groups**, not twelve independent experimental replicates. The original final models received 1,125 epochs / 18,000 updates each. Parameter counts and update counts were matched; computational cost was not guaranteed identical.

## Start locally

Use Python 3.12 and install a compatible PyTorch build for your machine. CPU is sufficient for the checks below. Run from the downloaded repository root:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m spatial_memory_kit verify --root .
python -m spatial_memory_kit replay --root . --model coupled_b7_s33101 --route a_one --state trained
python -m spatial_memory_kit replay --root . --model coupled_b7_s33101 --route a_two --state trained
python -m spatial_memory_kit replay --root . --model coupled_b7_s33101 --route a_two --state initial
python -m spatial_memory_kit replay --root . --model coupled_b7_s33101 --route a_two --state trained --history reset8
python -m spatial_memory_kit replay --root . --model coupled_b7_s33101 --route a_two --state trained --history reset-now
python -m spatial_memory_kit smoke-train --root . --output smoke-run
```

`verify` checks the packaged model/trace files. `replay` recomputes model outputs on a recorded trajectory; an initial-model run is a comparison with its paired starting weights. `smoke-train` demonstrates a small training execution. Its loss is an execution check, **not** a new spatial-memory result or a reproduction of the full study.

`--history full`, `reset8`, and `reset-now` compare full history, replay of at most the last eight observations from a reset state, and the current observation only. [The example comparison script](examples/compare_memory_controls.py) evaluates both trained/initial weights and all three history settings. [Saved release control results](examples/memory_control_results.json) cover 144 conditions: 12 models × 2 recorded routes × 2 weight states × 3 history settings. Each condition uses one selected episode, not 49 independent samples or the full H2 evaluation.

Release CPU checks reproduced all 24 trained traces with maximum absolute error `5.692243576049805e-6` (tolerance `1e-5`). Every exported tensor matched its source exactly. The [40-update smoke reference](examples/smoke_reference.json) reduced validation loss from approximately 0.157437 to 0.076054, but did not beat the current-image-copy baseline at either horizon. It supports execution, not spatial-memory achievement.

The featured model `coupled_b7_s33101` was selected for explanation. It is not advertised as the best model. Use the other conditions and seeds before drawing an architectural conclusion.

## Read a trajectory correctly

At time `t`, `h_t` incorporates observations only through `t`. In the coupled condition, `p_t` is computed using the previous action and can enter the observation update. `z_1(t)` and `z_5(t)` imagine future states using the specified future **actions**, without future observations. Future images are targets for evaluation/display, not inputs to the recurrent update.

At the true episode start, the previous-action vector is an all-zero sentinel. The coupled prior is computed but is **not adopted**. In the separate condition, an observation-update prior is not computed. The masks in each trace preserve this distinction.

A grid of 128 activation values is an arrangement of feature indices, not a physical circuit, a decoded map, or evidence that a particular unit represents a place. The same index need not mean the same thing in different models.

## Result to keep in mind

For the same featured checkpoint, the original full evaluation reported native future-image candidate accuracy of **100% / 100%** at horizons 1 / 5 on route A1, falling to **42.1875% / 37.5%** on the rearranged route A2. Resetting state before the last eight observations gave **25% / 25%** on A2. History still contributed, but robust route transfer was not established.

These percentages come from the original multi-case evaluation. The 24 included trajectories are **one preselected case per model/route**, and cannot reproduce those aggregate percentages by themselves. The release contains the evidence summary and clearly limits what its replay commands reproduce. See [results](docs/results.md).

[Saved source evaluation values](docs/evidence/saved_results.json) include the original 36-candidate A1/A2 aggregates and v33 gates, with source digests and JSON pointers. These are exported saved values, not newly recomputed full-study results; the packaged weight inventory remains 24 v33 trained/initial models.

## Reproduction boundary

This is a focused v0.1 release of the final v33 comparison and trace-replay tooling. It does not ship or retrain all 39 historical candidates, all training datasets, all task variants, or the complete original evaluation pipeline. It does not claim independent laboratory replication, external peer review, causal circuit identification, autonomous exploration, or long-term survival.

The static companion is a **recorded replay**, not live inference or a freely navigating agent. The Python kit recomputes the forward pass. The release does not include the long narrated project video or third-party speech-model weights.

## Files and provenance

```text
models/                 trained and initial safetensors, configs, manifest
traces/                 numeric NPZ traces and manifest
src/spatial_memory_kit/  portable model and command-line tools
docs/                   methods, results, release scope, source-record hashes
examples/               executable usage examples
```

Safetensors stores the original FP32 tensors without changing their values. JSON architecture files describe the custom PyTorch network; these are not Transformers checkpoints. Source checkpoint digests and release digests identify different representations of the same tensor contents.

## License and attribution

Original code, model weights, generated numerical traces, and documentation are released under the [MIT License](LICENSE). Dependencies retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Please cite this repository and a release identifier or commit when reporting results; [CITATION.cff](CITATION.cff) provides citation metadata.
