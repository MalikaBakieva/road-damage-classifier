# Dataset — RDD2022

## Source

**RDD2022 — Road Damage Dataset 2022**
Repository: <https://github.com/sekilab/RoadDamageDetector>
Paper: Arya, D., Maeda, H., Ghosh, S.K., Toshniwal, D., Sekimoto, Y. (2022).
*RDD2022: A multi-national image dataset for automatic road damage detection.*

**Licence:** CC BY-SA 4.0 — free to use and adapt with attribution, derivative
works share-alike. Compatible with this prototype's use.

**Scale:** ~47,000 road images, 55,000+ bounding-box annotations, six countries
(Japan, India, Czech Republic, Norway, United States, China).

This repository uses **Japan, India and Norway**: Japan for volume and label
quality, India for unpaved and heavily degraded surfaces, Norway for northern
lighting and winter conditions. Change `data.countries` in the config to adjust.

## Citation

```bibtex
@article{arya2022rdd2022,
  title   = {RDD2022: A multi-national image dataset for automatic road damage detection},
  author  = {Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar
             and Toshniwal, Durga and Sekimoto, Yoshihide},
  journal = {arXiv preprint arXiv:2209.08538},
  year    = {2022}
}
```

## Damage codes

| Code | Meaning | Our class |
|---|---|---|
| D00 / D01 | Longitudinal crack (wheel mark / construction joint) | `longitudinal_crack` |
| D10 / D11 | Transverse crack | `transverse_crack` |
| D20 | Alligator / fatigue crack | `alligator_crack` |
| D40 | Pothole, rutting, bump, separation | `pothole` |
| D43 | Crosswalk blur | *excluded — not damage* |
| D44 | White-line blur | *excluded — not damage* |
| D50 | Utility hole | *excluded — not damage* |

D43/D44/D50 describe marking wear and street furniture, not structural road
damage, so including them in a damage taxonomy would misrepresent the output.

## Download

The dataset is ~12 GB. It is **not** committed to this repository (`data/` is
git-ignored) — never commit dataset copies.

```bash
# 1. Download RDD2022.zip from the official repository above
# 2. Unpack into data/raw/ so the tree looks like:

data/raw/
├── Japan/train/images/*.jpg
├── Japan/train/annotations/xmls/*.xml
├── India/train/images/*.jpg
├── India/train/annotations/xmls/*.xml
└── Norway/train/...
```

The parser also accepts `data/raw/<Country>/images` + `annotations/xmls`
(without the `train/` level) and `annotations/` without the `xmls/` level,
because public mirrors differ.

No dataset? Run `python scripts/make_sample_data.py` to generate a synthetic
tree with the same structure — enough to exercise the whole pipeline, not enough
to produce a meaningful model.

## Annotation format

PASCAL VOC XML, one file per image:

```xml
<annotation>
  <filename>Japan_000000.jpg</filename>
  <size><width>600</width><height>600</height><depth>3</depth></size>
  <object>
    <name>D00</name>
    <bndbox><xmin>274</xmin><ymin>452</ymin><xmax>412</xmax><ymax>600</ymax></bndbox>
  </object>
</annotation>
```

Images with no `<object>` element are genuine negatives and are kept — dropping
them would remove the cleanest source of `not_damaged` samples.

## Preprocessing decisions

Run with `make prepare`. Each decision and its reason:

1. **Crop extraction.** Each annotated box is expanded by 25 % on every side and
   saved as a classification sample. *Why:* a single photo often contains
   several damage types, so a whole-image label is ambiguous; the context margin
   is included because a crack cropped to its exact bounding box loses the
   surrounding road surface that makes it interpretable.

2. **Negative sampling.** Up to `negatives_per_image` square patches are sampled
   from regions overlapping no annotation, restricted to the **lower 65 % of the
   frame**. *Why:* uniform sampling would produce sky and building crops, making
   `not_damaged` trivially separable and the reported score meaningless.

3. **Minimum box size 24 px.** Smaller boxes are dropped. *Why:* after resizing
   to 224×224 they contain no recoverable signal and act as label noise.

4. **Degenerate boxes dropped.** Boxes where `xmax <= xmin` or `ymax <= ymin` —
   a known RDD2022 data-quality issue — are discarded.

5. **Corrupt files tolerated.** Unparsable XML and truncated JPEGs are logged
   and skipped rather than raising. *Why:* one bad file in 47,000 must not kill a
   multi-hour job.

6. **RGB conversion.** All crops are converted to RGB; greyscale and RGBA inputs
   are normalised at both training and inference time.

7. **Image-level splitting, stratified by dominant class, 70/15/15, seed 42.**
   *Why:* see `docs/DISCOVERY.md` — this is the primary leakage control, and it
   is enforced by a unit test.

8. **ImageNet normalisation.** Required by the pretrained backbones; the exact
   same transform is used by training, evaluation and the API, so serving cannot
   silently diverge from training.

## Outputs

| File | Contents |
|---|---|
| `data/interim/annotations.csv` | Object-level table parsed from XML |
| `data/interim/crops/<class>/*.jpg` | Extracted crops on disk |
| `data/processed/crops.csv` | Crop index with labels and split assignment |
| `data/processed/dataset_stats.json` | Counts per class, per split, per country |

## Data-quality issues observed

- Severe class imbalance (potholes ~8–10× rarer than longitudinal cracks).
- Annotation consistency varies by country; India is noisiest.
- Unannotated regions are not guaranteed damage-free — irreducible label noise
  in the negative class.
- Motion blur, low sun, wet reflections and shadows throughout.
- A small number of truncated JPEGs and malformed XML files.
