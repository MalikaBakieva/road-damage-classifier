# Model Card — Road Damage Classifier (GOV-01)

## Model details

| | |
|---|---|
| **Developed for** | Municipal Infrastructure Department (GOV-01 field scenario) |
| **Version** | 1.0.0 |
| **Type** | Image classification (CNN, transfer learning) |
| **Architecture** | ImageNet-pretrained backbone (`resnet18` reference; `efficientnet_b0` / `convnext_tiny` configurable) with a fresh classification head |
| **Input** | RGB image patch, 224×224, ImageNet-normalised |
| **Heads** | **Binary:** `not_damaged`, `damaged` (primary). **Multiclass:** `not_damaged`, `longitudinal_crack`, `transverse_crack`, `alligator_crack`, `pothole` (secondary) |
| **Training data** | RDD2022 (Japan, India, Norway), CC BY-SA 4.0 |
| **Framework** | PyTorch + timm |
| **Licence** | MIT (code); dataset CC BY-SA 4.0 |

## Intended use

**Intended:** ranking incoming road-condition photographs so inspectors review
the most likely and most severe damage first; reducing manual triage effort;
producing consistent first-pass labels across a growing image queue.

**Not intended, and must not be used for:**

- Engineering safety assessments or structural condition ratings
- Deciding repair budgets, liability or legal claims without human review
- Certifying that a road is safe — the model detects *visible surface damage*
  and cannot see sub-surface condition
- Any fully automated decision without a human in the loop

**Users:** municipal inspection staff and the triage system that queues their
work. Not intended for direct public consumption of raw predictions.

## Factors

Performance varies with **lighting** (daytime training data; degradation at
night and in low sun), **weather** (dry surfaces dominate; wet, snowy and leaf-
covered surfaces are under-represented), **camera geometry** (vehicle-mounted
dashcam framing; close-up handheld phone photos are out of distribution),
**image quality** (motion blur and low resolution hurt thin cracks most), and
**geography** (Japan/India/Norway road construction styles).

## Metrics

**Primary: macro-F1** — chosen because the classes are heavily imbalanced and
every damage type matters to the client. Accuracy is dominated by the majority
class: a model that never predicts `pothole` still scores ~90 % accuracy on the
binary task, which is precisely the failure the client cannot afford.

**Also reported:** per-class precision/recall/F1 with support; recall on
`damaged` (asymmetric cost — a missed defect is worse than a false alarm routed
to triage); balanced accuracy; ROC-AUC; row-normalised confusion matrix;
expected calibration error; and a coverage/accuracy threshold sweep.

**Evaluation data:** a held-out test split (15 %), partitioned at **image**
level so no photograph contributes crops to more than one split. The test split
is evaluated exactly once, after model selection on validation macro-F1.

Results are written to `reports/<task>/metrics.json`, with the confusion matrix
plot, classification report, threshold sweep and robustness table alongside.

> **Reporting note.** Numbers produced by `make demo` come from synthetic
> images and must never be quoted as model performance. Only fill the results
> table below from a run on real RDD2022 data.

### Results (fill in after `make evaluate` on real data)

| Metric | Binary | Multiclass |
|---|---|---|
| Macro-F1 | — | — |
| Accuracy | — | — |
| Balanced accuracy | — | — |
| `damaged` recall | — | n/a |
| ROC-AUC | — | — |
| ECE | — | — |

## Training procedure

Frozen-backbone linear probing for the first epoch to settle the randomly
initialised head, then full fine-tuning with AdamW, a 10× higher head learning
rate, cosine decay, gradient clipping at 1.0 and label smoothing 0.05. Class
imbalance is handled by inverse-frequency loss weights normalised to mean 1
(alternative: weighted sampler). Early stopping on validation macro-F1 with
patience 4; the best checkpoint by validation macro-F1 is the one saved.

Augmentation targets the documented failure modes: random-resized crop
(scale 0.7–1.0), horizontal flip, ±8° rotation, brightness/contrast/saturation
jitter, occasional Gaussian blur, random erasing.

All runs are logged to MLflow with the full config, per-epoch metrics and the
resulting artefacts.

## Ethical considerations

**Geographic and socioeconomic bias.** Training data is Japan/India/Norway.
Applied to a city with different asphalt, markings or climate, error rates will
differ — plausibly worse in neighbourhoods whose road surfaces are least similar
to the training distribution. Because the output drives inspection *priority*,
systematic under-detection in one area means slower repairs there. Mitigation:
report per-country metrics, re-validate on local imagery before deployment, and
keep low-confidence cases in a human queue.

**Privacy.** Street-level imagery can contain readable licence plates,
pedestrians and building frontages. RDD2022 is already public and this
repository does not redistribute it. A production deployment handling citizen
submissions would need face/plate blurring, a retention policy and a lawful
basis for processing.

**Over-trust.** The largest practical risk is staff treating a confident output
as an assessment. Mitigations: the `disclaimer` field on every response, the
`needs_human_review` flag below the confidence threshold, and explicit scope
language in the UI and documentation.

**Accountability.** Predictions are advisory. Repair and safety decisions remain
with qualified municipal staff.

## Caveats and recommendations

- Fine-tune on a few hundred locally collected, locally labelled photographs
  before deployment; this is expected to be the single largest improvement.
- Choose the confidence threshold from `reports/<task>/threshold_sweep.csv`
  together with the inspection team, based on their real review capacity.
- Route production decisions on the **binary** head; treat the multiclass output
  as advisory detail only above the confidence threshold.
- Re-evaluate periodically on fresh labelled samples; road appearance,
  camera hardware and submission channels all drift.
- The model has no "not a road" class and will confidently classify unrelated
  images. Validate upstream that submissions are road photographs.

See [`LIMITATIONS.md`](LIMITATIONS.md) for the full discussion.
