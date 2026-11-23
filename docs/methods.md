# Methods and interpretation

## Three different questions

1. **Prediction:** does a trained model predict the next image/body better than a simple current-observation copy?
2. **Memory use:** does a model's own prediction change meaningfully when earlier state is removed or replaced, under controlled observations?
3. **Transfer:** does the same representation or behavior remain useful when route, direction, or distance changes?

A positive answer to one question does not establish the others. An external position decoder is an analysis instrument, not part of the world model. Its failure can reflect a change of readout relationship rather than no information in the state.

## Model family

The visual encoder is two convolutions (1→16 and 16→32, kernel 3, stride 2, padding 1), followed by a 512→64 projection with SiLU activations. Visual features, five body values, and the four-way previous action form a 73-value observation input. The observation GRU has a 128-value hidden state. An action-only GRUCell takes four action values and a 128-value state. The decoder is 128→280→261, producing image, contact, motion, and energy outputs.

The four conditions cross observation update (`separate`, `coupled`) with action-transition initial update-gate bias (2, 7). The `coupled` model uses the action transition for the observation prior as well as imagined future dynamics. The `separate` model keeps observation recurrence independent of this prior. All models have the same parameter count; sharing use of a component changes computations, not the number of stored parameters.

The three paired seed groups are 33101, 33113, and 33127. Initial weights are released so that history retention by an untrained recurrent network can be measured directly. Matching seeds does not mean every condition's parameters are numerically identical: the deliberately changed gate-bias initialization remains different.

## Temporal semantics

`h_t` uses `o_0…o_t`. The prior `p_t` uses `a_(t−1)`. Imagined state `z_k(t)` uses `a_t…a_(t+k−1)` and no future observations. The all-zero previous-action vector at the start marks absence of an earlier action. A computed but unadopted prior must not be displayed as if it influenced the first observation update. Separate-model priors are unavailable.

The saved traces contain 49 observed states and longer body/action/target arrays needed for future targets. A reader must crop observation inputs to the recorded observation times and must never feed target observations into the history. Observation images in the exported traces are already normalized float32 values; dividing them by 255 again is incorrect.

## Recorded trace selection

All 12 trained final v33 models have traces for A1 (`a_one`) and A2 (`a_two`). The source recording selected case index 0 and query index 48 before producing the explanatory display. Thus all 24 files are included, while each remains only one illustration from a larger evaluation. The featured model is chosen for explanation, not ranked as the best checkpoint.

Recorded tensors include observed state `h`, action prior, imagined `z1`/`z5`, current images, future targets, predictions, and prior-computation/adoption masks. Simulator display coordinates are evaluation metadata, not model inputs. The trace source used FP32 with TF32 disabled. Portable CPU replay reports actual deviation from those records and a stated tolerance; exact tensor conversion does not imply bit-identical arithmetic on every runtime.

## What a replay proves

The command recomputes a saved trajectory with an identified model and can compare it to a saved trained-model trace. A successful comparison supports faithful loading and forward-pass reproduction for those inputs. It does not rerun all cases, refit every external probe, establish all aggregate percentages, or independently reproduce the training process.

The training smoke run demonstrates that a small optimization can execute. A falling loss alone is not evidence of spatial memory, better-than-copy prediction, route transfer, or survival. Its separate output directory keeps it distinct from the published final weights.

## Display interpretation

The static companion animates recorded values. Its 8×16 heatmap is an index layout for a 128-component vector. It is not a physical neural layout or a decoded environment map. Arrows in the implementation diagram describe computational data flow, not causally identified biological circuits. Feature indices cannot be assumed to align semantically across checkpoints.
