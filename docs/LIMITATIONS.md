# Limitations, Risks and Next Steps

> Brief §9 (acceptance criterion: "known limitations, risks and important
> assumptions are documented") and §11 (questions that must be resolved).

---

## 1. Answers to the questions in §11

### Which damage classes are realistic given the dataset and timeframe?

Four damage classes plus a negative class. RDD2022's internationally consistent
codes are D00 (longitudinal crack), D10 (transverse crack), D20 (alligator
crack) and D40 (pothole); these are the only ones annotated across all
countries with enough volume to train and, more importantly, enough volume to
*evaluate*. D43/D44/D50 (crosswalk blur, white-line blur, utility hole) are not
damage and are excluded. Country-specific codes were dropped because a class
with a few dozen test examples produces an F1 estimate too noisy to defend.

Realistic outcome ordering, and why:

- **Binary (`damaged`/`not_damaged`)** — reliable. Damage-vs-clean is a large
  visual difference and both classes have thousands of samples.
- **Potholes** — separable. Dark, filled, high-contrast, visually distinct.
- **Alligator cracking** — moderately separable. Distinctive mesh texture.
- **Longitudinal vs transverse cracks** — the weak point. They differ mainly by
  *orientation*, and orientation is exactly what horizontal-flip and rotation
  augmentation partially destroys. Expect most multiclass confusion here.

### How will class imbalance affect training and evaluation?

Training: with no correction, gradient descent optimises the majority class and
the model learns to under-predict potholes — the class the client most needs.
Handled by inverse-frequency loss weighting (default, `imbalance_strategy:
weighted_loss`) with a weighted sampler as a config-selectable alternative.
Weights are normalised to mean 1 so the loss scale, and therefore the learning
rate, stays comparable across strategies.

Evaluation: this is the bigger danger. Accuracy is dominated by the majority
class — a model that never predicts `pothole` still scores ~90 % accuracy on
the binary task. That is why **macro-F1 is the primary metric** and the metric
used for early stopping and model selection; `tests/test_model.py::
test_macro_f1_punishes_ignoring_a_minority_class` encodes this as a test. The
row-normalised confusion matrix is reported alongside so *which* classes fail is
visible, not just that something failed.

Note the imbalance is also *evaluation* imbalance: with few pothole examples in
the test split, per-class F1 for potholes has wide error bars. Grouped k-fold CV
is the honest fix if a confidence interval is needed.

### What image variations may cause the model to fail?

Measured explicitly by `evaluate.py::robustness_check`, which re-scores the test
split under simulated shifts and writes `reports/<task>/robustness.json`:

| Shift | Simulation | Expected effect |
|---|---|---|
| Low light / night | brightness ×0.4 | Large drop — training data is daytime |
| Glare / overexposure | brightness ×1.6 | Moderate drop |
| Low contrast (fog, overcast) | contrast ×0.45 | Moderate drop |
| Motion blur | Gaussian blur σ=2.0 | Large drop for thin cracks |
| Off-axis camera | rotation 15° | Large drop for longitudinal/transverse |
| Low resolution | 4× downscale + upscale | Large drop for hairline cracks |

Additional failure modes not simulated but expected in the field: wet surfaces
(reflections mimic cracks), snow or leaf cover (occludes damage entirely), tar
sealant repairs (dark linear marks read as cracks), shadows from trees and
poles (linear dark features), extreme close-ups without road context, and
photos that are not of roads at all — the model has no "reject" class and will
confidently classify a photo of a wall.

### Would a binary damage/no-damage scope be more reliable than many classes?

**Yes, and that is why binary is the primary scope.** Three reasons: (1) the
classes are far more separable, so macro-F1 is materially higher; (2) it removes
the longitudinal/transverse confusion that dominates multiclass error; (3) it
matches the department's actual decision — reports are triaged for *inspection*,
and an inspector determines the damage type on site far more reliably than a
model does from one photo.

The multiclass head is shipped because knowing "pothole" versus "hairline crack"
genuinely changes queue priority, and it is honest to show what the finer scope
can and cannot do. Production recommendation: **route on the binary output, use
the multiclass output only as advisory detail, and only when its confidence
exceeds the threshold.**

### What metric best reflects performance across all classes?

**Macro-F1.** It averages per-class F1 with equal weight, so a class the model
ignores drags the score down regardless of how rare it is. Accuracy and micro-F1
are both dominated by the majority class and would hide exactly the failure the
client cares about. Balanced accuracy is a reasonable alternative but ignores
precision, which matters because false alarms consume inspector time.

Reported alongside macro-F1: per-class precision/recall/F1 with support, recall
on `damaged` (asymmetric cost — a missed defect is worse than a false alarm),
ROC-AUC, expected calibration error, and a coverage/accuracy threshold sweep so
the operating point is an evidence-based choice rather than a default.

### How will you test images that differ from the training distribution?

Three layers, in increasing order of realism:

1. **Synthetic corruption of the test split** (implemented) — the robustness
   table above. Cheap, repeatable, runs in CI.
2. **Cross-country hold-out** (supported) — train on Japan+India, evaluate on
   Norway by setting `data.countries`. Norway differs in climate, lighting and
   road construction, so this is a genuine domain-shift test rather than a
   simulated one.
3. **Real out-of-distribution collection** (recommended before deployment) —
   200–500 photographs taken by the department's own inspectors and citizens on
   local roads, hand-labelled, used as a *held-out acceptance set*. This is the
   only measurement that answers the question the client actually has.

Operationally, the confidence threshold is the runtime defence: inputs unlike
anything seen in training tend to produce low max-probability, which flags
`needs_human_review` instead of a confident wrong answer. This is a mitigation,
not a guarantee — neural networks can be confidently wrong on OOD input, which
is why layer 3 matters.

---

## 2. Known limitations

**Data**

- Trained on RDD2022 only: daytime, vehicle-mounted, Japan/India/Norway.
- Unannotated regions used as negatives may contain missed damage — irreducible
  label noise that caps achievable recall.
- Consecutive dashcam frames can show the same defect; image-level splitting
  removes crop-level leakage but not all near-duplicate-frame leakage.
- Annotation consistency varies by country; India's labels are noticeably
  noisier than Japan's.

**Model**

- Classifies **visible surface damage only**. It cannot assess depth, structural
  severity, sub-surface condition or repair urgency.
- No "not a road" rejection class — it will classify any image confidently.
- Single-label per crop: a patch containing both a pothole and cracking gets one
  label.
- Trained at 224×224; hairline cracks below a few pixels after resizing are
  effectively invisible.
- Calibration is imperfect (see ECE in `reports/*/metrics.json`); the confidence
  score is a useful ranking signal, not a true probability.

**System**

- CPU inference is ~40–80 ms per image — fine for triage, not for real-time
  video.
- No authentication, rate limiting or audit logging: prototype scope, explicitly
  out of scope per the brief.
- Uploads are processed in memory and not persisted; there is no feedback loop
  capturing inspector corrections.

---

## 3. Risks

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Over-reliance on model output as a safety judgement | High | Medium | Disclaimer on every response, "triage support" language in UI and docs, human-review flag |
| Domain shift on local imagery | High | High | Robustness evaluation, documented fine-tuning path, OOD acceptance set before deployment |
| Missed damage (false negatives) | High | Medium | `damaged` recall reported separately; threshold tunable toward recall |
| False alarms flooding inspectors | Medium | Medium | Threshold sweep gives the coverage/precision trade-off explicitly |
| Geographic bias in service levels | High | Medium | Per-country metrics; re-validate on local data; keep human review |
| Privacy in street-level photos | Medium | Medium | Dataset not redistributed; blurring and retention policy required for production |
| Silent model degradation over time | Medium | High | MLflow run history; recommend periodic re-evaluation on fresh labelled samples |

---

## 4. Recommended next steps

**Before any deployment**

1. Collect and label 200–500 photographs from the department's own inspectors
   and citizen channel; evaluate the current model on them untouched.
2. Fine-tune on that local set and re-evaluate — expect the largest single gain
   here, larger than any architecture change.
3. Choose the confidence threshold from `threshold_sweep.csv` together with the
   inspection team, based on their real capacity for human review.

**Model improvements, in expected order of value**

4. Compare `efficientnet_b0` and `convnext_tiny` against the `resnet18`
   reference (one config change each; log all three in MLflow).
5. Add grouped 5-fold CV to attach confidence intervals to macro-F1.
6. Add temperature scaling on the validation split to improve calibration — this
   directly improves the threshold's usefulness.
7. Test-time augmentation (horizontal flip) for a small, cheap accuracy gain.
8. Consider a severity/extent regression head if the department can supply
   severity labels.

**System**

9. Persist predictions and inspector outcomes to build a feedback dataset — this
   is what makes version 2 meaningfully better than version 1.
10. Add authentication, rate limiting and structured audit logging.
11. Add drift monitoring on the input distribution and on the rate of
    low-confidence predictions.
