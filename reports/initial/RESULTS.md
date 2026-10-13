# Real detector diagnostic run

Date: 2026-10-13. 64 COCO val2017 images, ordered by image ID. 2 repeats. Seed 20261013.

This is fixed-threshold detection analysis, **not COCO mAP**. Small means original bbox area < 32² pixels. No crowd-ignore metadata is available in YOLO labels.

| Strategy | Calls/image | Precision50 | Recall50 | Small recall50 | TP / FP / FN | Median ms |
|---|---:|---:|---:|---:|---|---:|
| global | 1.0 | 72.51% | 54.55% | 23.13% | 240 / 91 / 200 | 12.4 |
| uniform | 4.0 | 42.39% | 58.86% | 31.97% | 259 / 352 / 181 | 41.0 |
| random | 4.0 | 44.24% | 59.32% | 30.61% | 261 / 329 / 179 | 40.7 |
| adaptive | 4.0 | 45.35% | 62.05% | 36.05% | 273 / 329 / 167 | 40.8 |
| evidence | 4.0 | 45.08% | 61.36% | 36.05% | 270 / 329 / 170 | 41.0 |
| texture | 4.0 | 44.72% | 60.68% | 33.33% | 267 / 330 / 173 | 41.4 |
| full | 10.0 | 31.57% | 63.64% | 39.46% | 280 / 607 / 160 | 99.0 |

## Reading the evidence

Compare adaptive against uniform/random/evidence/texture at equal calls. Full is an exhaustive slicing reference with a larger budget. Inspect per-image CSV and trace JSON before claiming benefits.

Paired image-bootstrap 95% CI for adaptive minus uniform small recall: [-0.020202020202020204, 0.09524407753050965] (fractions, 2,000 resamples).

## Limits

- Deterministic small validation subset, not representative of full COCO
- Pretrained YOLOv8 weights; no new detector training
- Tile count is a budget proxy; latency also includes content-dependent overhead
- Two timing repeats are diagnostic; repeat across devices for performance claims
- Heuristic confidence/texture scores are not calibrated uncertainty

Exact weights, dependencies, input hashes, and configuration are in summary.json.
