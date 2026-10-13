# Real detector diagnostic run

Date: 2026-10-13. 64 COCO val2017 images, ordered by image ID. 2 repeats. Seed 20261013.

Internal crop-edge filtering margin: 0.02. This is fixed-threshold detection analysis, **not COCO mAP**. Small means original bbox area < 32² pixels. No crowd-ignore metadata is available in YOLO labels.

| Strategy | Calls/image | Precision50 | Recall50 | Small recall50 | TP / FP / FN | Median ms |
|---|---:|---:|---:|---:|---|---:|
| global | 1.0 | 72.51% | 54.55% | 23.13% | 240 / 91 / 200 | 12.3 |
| uniform | 4.0 | 65.81% | 58.18% | 31.29% | 256 / 133 / 184 | 41.7 |
| random | 4.0 | 67.45% | 58.41% | 30.61% | 257 / 124 / 183 | 41.1 |
| adaptive | 4.0 | 65.52% | 60.45% | 34.69% | 266 / 140 / 174 | 41.1 |
| evidence | 4.0 | 65.20% | 60.45% | 36.05% | 266 / 142 / 174 | 41.1 |
| texture | 4.0 | 66.75% | 59.77% | 31.97% | 263 / 131 / 177 | 41.3 |
| full | 10.0 | 59.78% | 62.50% | 38.78% | 275 / 185 / 165 | 99.8 |

## Reading the evidence

Compare adaptive against uniform/random/evidence/texture at equal calls. Full is an exhaustive slicing reference with a larger budget. Inspect per-image CSV and trace JSON before claiming benefits.

Paired image-bootstrap 95% CI for adaptive minus uniform small recall: [-0.023809523809523808, 0.08451557831839517] (fractions, 2,000 resamples).

## Limits

- Deterministic small validation subset, not representative of full COCO
- Pretrained YOLOv8 weights; no new detector training
- Tile count is a budget proxy; latency also includes content-dependent overhead
- Timing repeats are diagnostic; repeat across devices for performance claims
- Heuristic confidence/texture scores are not calibrated uncertainty

Exact weights, dependencies, input hashes, and configuration are in summary.json.
