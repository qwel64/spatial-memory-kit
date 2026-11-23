# Results, limits, and release evidence

The source study is complete as an initial bounded comparison. The integrated artificial-organism capability was not achieved. This release preserves that distinction.

## Original aggregate results relevant to this release

| Comparison | Original finding | What it does not establish |
|---|---|---|
| Final v33 combined criterion: short-task retention, long wait, movement-native prediction, and spatial readout | 0 of 12 final candidates satisfied the complete fixed criterion | That all models learned and then forgot; acquisition preconditions must be checked |
| Featured `coupled_b7_s33101`, A1 | Native image-candidate accuracy 100% / 100% at h1 / h5 | General navigation or success on other routes |
| Same checkpoint, A2 rearranged route | 42.1875% / 37.5%; reset-eight 25% / 25% | Stable route-independent spatial understanding |
| Source-fixed position readout transferred A1→A2 | x and y R² negative for all 36 original ABC candidates | No position information anywhere in the state |
| Featured checkpoint recalibrated within A2 | x/y R² approximately 0.705 / 0.947 on held-out energy samples | Transfer to an additional unseen route |
| Bias-7 coupled minus separate A2 h1 accuracy | +17.1875, 0, −6.25 percentage points across the three paired groups | A consistent architectural advantage |

These are saved results of the full original evaluation. This focused kit includes 24 illustrative trajectories and 24 v33 trained/initial models. It **does not** include all cases and probes needed to recompute the aggregate table. The table is a provenance-backed summary, distinct from the replay checks that a user can execute directly.

[Machine-readable saved results](evidence/saved_results.json) include the original 36-candidate A1/A2 aggregate values and v33 gates. Their source digests and JSON pointers permit checking the exported figures. These are extracted saved results, not a newly executed full evaluation. [The release's separate 144 control runs](../examples/memory_control_results.json) use only the two selected trajectories per v33 model and have a narrower interpretation.

The original ABC set contained 36 candidates and the broader final comparison contained 39. Do not treat the v0.1 inventory as all 39 models, or treat rows from different model generations as abilities of one integrated agent.

## Why the negative results matter

In the featured model, history remains useful on A2, but accuracy declines when the route changes. An untrained recurrent network can also preserve action-history information; trained-versus-initial comparisons are necessary to attribute a benefit to prediction training. Position-readout transfer, within-task position readout, and the model's own use of state in prediction are separate measurements.

The original direction test separated inward-facing ambiguous cases (four alternatives) from sideways cases with one possible future image. Counting the one-alternative cases as memory success would inflate the conclusion. The original distance test also showed that a high R² or good image choice can coexist with poor tracking of distance change. Neither broad generalization nor long-term autonomous survival was demonstrated.

## Numerical qualification of the broader historical project

The original project's separate D food-choice evaluation is outside this release's executable scope. That evaluation used FP64 brain forward passes and FP32 head/normalization training, unlike the ABC FP32 trace context. A high-precision normalization diagnostic still failed two scale components for `coupled_b7_s33101` reset8-fit. The acceptance scope was explicitly changed to verify the declared train-only CUDA FP32 normalization procedure and replay with saved normalization constants and fixed heads. This is not a claim that all original precision checks passed or that results are invariant to a different normalizer. The audit did not independently rerun all 631,800 head-training updates.

## Source records

The following digests identify original records used to write this release. Paths are identifiers relative to the original project archive, not downloadable links in this focused kit.

| Original record | SHA-256 |
|---|---|
| `research/PROJECT_FINAL_INTERPRETATION.json` | `04475710cb42014b7a42333c8b9d9dc6a5d0531d1ec623bd92ed9e23d0cd6798` |
| `research/TERMINAL_ABC_INTERPRETATION.json` | `d92bcecb3ecf14800f6a43a90f1a52b00c04bfc9da3d91cad8a5930e03808fe9` |
| `reports/project_final/same_brain_matrix.json` | `c10350f4546df221efbb5bf17accf6742d3f7ebcb2d1cf8df2eeaac6cebd6118` |
| `reports/project_final_video/measured_trace/manifest.json` | `4b6ece21aea7f84616d08234b13e833d49384024f89c4d21f45727a5f8de048a` |

The first record's `/sections/same_brain_findings/text` is the source of the aggregate descriptions above; `/sections/limits_and_judgment/text` documents the bounded interpretation. The release manifests record individual checkpoint and trace provenance. Packaging verification is not external peer review or an independent laboratory replication.
