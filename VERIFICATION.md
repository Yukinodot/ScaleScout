# Verification record

Checked locally on 2026-10-13.

| Item | Evidence | Status |
|---|---|---|
| Core geometry, budget, NMS, matching, edge filtering | `python -m pytest -q` | 22 passed |
| Python syntax | `python -m compileall -q src scripts` | Passed |
| JavaScript syntax | Node `--check` on app.js and preview-data.js | Passed after embedding report data |
| CLI on CPU | COCO image 000000000139.jpg, K=3 | 17 detections, 4 calls, JSON and JPEG produced |
| Real GPU evaluation | Initial and refined 64 images, then 64 nonoverlapping holdout images | Completed; 7 strategies, 2 repeats each |
| Browser result viewer | Strategy switch, rendered tables/canvas, local original-image import and matching hash | Checked in Edge |
| Course training script | `--help` and syntax only | 300-epoch training not executed |
| YOLO-World example | `--help` and syntax only | Actual model inference not executed |
| Official COCO mAP | Requires official metadata and evaluator | Not implemented or claimed |

The reports preserve unsuccessful comparisons. Adaptive versus uniform small-recall bootstrap intervals contain zero. The evidence-only ablation has slightly higher small recall than adaptive in both refined subsets.

GitHub Actions runs lightweight core tests on push/PR. Its remote execution status must be checked separately; a local pass does not establish a remote CI pass.
