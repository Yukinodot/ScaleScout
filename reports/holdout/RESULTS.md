# Real detector diagnostic run

Date: 2026-10-13. 64 COCO val2017 images, ordered by image ID. 2 repeats. Seed 20261013.

Internal crop-edge filtering margin: 0.02. This is fixed-threshold detection analysis, **not COCO mAP**. Small means original bbox area < 32² pixels. No crowd-ignore metadata is available in YOLO labels.

| Strategy | Calls/image | Precision50 | Recall50 | Small recall50 | TP / FP / FN | Median ms |
|---|---:|---:|---:|---:|---|---:|
| global | 1.0 | 73.51% | 53.93% | 20.34% | 247 / 89 / 211 | 12.5 |
| uniform | 4.0 | 64.59% | 58.95% | 33.90% | 270 / 148 / 188 | 41.9 |
| random | 4.0 | 65.74% | 56.99% | 28.81% | 261 / 136 / 197 | 41.5 |
| adaptive | 4.0 | 62.33% | 59.61% | 36.44% | 273 / 165 / 185 | 41.8 |
| evidence | 4.0 | 62.50% | 60.04% | 37.29% | 275 / 165 / 183 | 41.6 |
| texture | 4.0 | 64.37% | 59.17% | 34.75% | 271 / 150 / 187 | 41.7 |
| full | 10.0 | 58.20% | 62.01% | 41.53% | 284 / 204 / 174 | 98.0 |

## Reading the evidence

Compare adaptive against uniform/random/evidence/texture at equal calls. Full is an exhaustive slicing reference with a larger budget. Inspect per-image CSV and trace JSON before claiming benefits.

Paired image-bootstrap 95% CI for adaptive minus uniform small recall: [-0.011111111111111112, 0.0660377358490566] (fractions, 2,000 resamples).

## Limits

- Deterministic small validation subset, not representative of full COCO
- Pretrained YOLOv8 weights; no new detector training
- Tile count is a budget proxy; latency also includes content-dependent overhead
- Timing repeats are diagnostic; repeat across devices for performance claims
- Heuristic confidence/texture scores are not calibrated uncertainty

Exact weights, dependencies, input hashes, and configuration are in summary.json.
