# Model card: Spatial Memory Kit v0.1

## Model details

- **Type:** custom recurrent sensory world model; not a language model.
- **Size:** 276,501 learned parameters per model; 18 FP32 tensors; recurrent state width 128.
- **Release family:** final v33, four architectural/initialization conditions and three paired seeds, with matched initial weights.
- **Owner:** qwel64; Hugging Face publishing identity nao1357.
- **License:** MIT for original release content.

This exploratory personal project used AI agents for implementation, analysis, checking, and writing. Agent-assisted auditing does not constitute external peer review or independent laboratory replication.

The input is a 16×16 grayscale observation, five body values (two contact signals, two motion values, one energy value), and four-way previous-action encoding. The encoder produces 64 visual features. The observation GRU consumes 73 input values and maintains 128 state values. An action-only GRUCell supports imagined rollouts. A width-280 decoder outputs 256 image values, two contact logits, two motion values, and one energy value. Coordinates, map labels, and route IDs are excluded from brain inputs and training teachers.

`separate` uses a separate observation update. `coupled` uses the action transition as a prior before observation correction when a previous action is available. `b2` and `b7` label the action-transition update-gate bias initialization; observation memory bias is 4. At a true episode start, the zero previous-action sentinel prevents adoption of the computed coupled prior.

## Intended use

Inspect small recurrent representations, compare paired trained/initial models, recompute recorded forward passes, and explore why sensory prediction, history use, and transfer of position readouts are different properties. The release is suitable as a bounded research and teaching example.

## Out-of-scope uses

No claims of deployment-ready robotic navigation, general spatial cognition, biological realism, consciousness, physical neural circuit reconstruction, autonomous foraging, or long-term survival. The demonstration viewer is a recorded replay; it does not accept arbitrary scenes or control an agent.

## Training and data

The original models were trained in a synthetic visual/body/action environment to predict future sensory/body observations. The final v33 checkpoints each record 1,125 epochs / 18,000 updates. The comparison pairs conditions within seeds 33101, 33113, and 33127. Between seed groups both initialization and sampled training energy differ; variation cannot be attributed to initialization alone.

This release includes the weights and a limited diagnostic trace subset, not the complete original training datasets or full training pipeline. The included training smoke command is an execution demonstration, not full v33 reproduction.

## Evaluation

The original study tested native image predictions, external position readouts, state resets/exchanges, and route/direction/distance changes. A featured model fell from A1 native image-candidate accuracy 100% / 100% to A2 42.1875% / 37.5% (horizons 1 / 5); A2 reset-eight baselines were 25% / 25%. The original combined criterion (short-task retention, long wait, movement-native prediction, and spatial readout) was unmet for all 12 final v33 candidates. The featured model had not satisfied the earlier acquisition precondition, so this is not evidence that it learned and subsequently forgot that ability.

These are summaries of the original full evaluation, not scores computed from the 24 packaged single-case trajectories. See [results and source digests](docs/results.md). The kit's replay check reports numerical agreement on the included traces, not a rerun of all aggregate results.

## Limitations

Three paired seed groups are insufficient to declare a universal architectural winner. Initial random recurrence can retain useful history. Failure of a source-trained linear readout under a distribution shift does not prove absence of information. Static activation displays identify neither causal units nor a physical map. Data are synthetic, finite, and diagnostic; generalization beyond tested tasks is unestablished.

The full historical project also included a separate food-choice evaluation with a changed numeric acceptance scope after two normalization components missed the original high-precision diagnostic. That pipeline and its trained heads are outside this v0.1 release; no claim that every original precision check passed is made. The original audit reproduced saved records and fixed-model calculations, not an independent rerun of all historical optimization updates.

## Format and reproducibility

Weights are safetensors and architecture/training metadata are JSON. Original source-checkpoint hashes and the hashes of exported files are distinct. Conversion must preserve every tensor value exactly. The portable CPU implementation and saved GPU traces can differ within the reported replay tolerance; file conversion equality and cross-runtime floating-point agreement are separate checks.

Use the manifest and command output to identify the model, initial/trained state, route, and numerical result. A model filename alone does not establish a general capability.
