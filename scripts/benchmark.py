"""Paired real-detector benchmark with per-image evidence and ablations."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import numpy as np
from scalescout.cli import load_image
from scalescout.detector import YOLODetector
from scalescout.metrics import read_yolo_labels, counts, summarize
from scalescout.pipeline import Config, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--budget", type=int, default=3)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--boundary-margin", type=float, default=0.02)
    p.add_argument("--output", type=Path, default=Path("reports/latest"))
    args = p.parse_args()
    if args.limit < 1 or args.repeats < 1:
        p.error("limit and repeats must be positive")
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    images = sorted(x for x in args.images.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"})[:args.limit]
    if not images:
        raise ValueError("No images found")
    missing = [x.stem for x in images if not (args.labels/(x.stem+".txt")).exists()]
    if missing:
        raise ValueError(f"Missing labels (never silently treated as negatives): {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    detector = YOLODetector(args.weights, args.imgsz, args.device)
    detector.warmup()
    strategies = ["global", "uniform", "random", "adaptive", "evidence", "texture", "full"]
    rows, timings = [], {s: [] for s in strategies}
    chooser = random.Random(args.seed)
    records = []
    for image_idx, path in enumerate(images):
        image = load_image(path)
        label_path = args.labels/(path.stem+".txt")
        labels = read_yolo_labels(label_path, *image.size)
        records.append({"image": path.name, "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(), "size": image.size})
        order = list(strategies)
        chooser.shuffle(order)
        for strategy in order:
            config = Config(strategy=strategy, budget=args.budget, seed=args.seed+image_idx,
                            boundary_margin=args.boundary_margin)
            traces = []
            for _ in range(args.repeats):
                predictions, trace = run(image, detector, config)
                detector.synchronize()
                timings[strategy].append(trace["timing_ms"]["total"])
                traces.append(trace)
            traces[0]["image_name"] = path.name
            traces[0]["input_sha256"] = records[-1]["image_sha256"]
            c = counts(predictions, labels)
            # Keep all selection traces. Do not let labels enter the pipeline.
            (args.output/f"{path.stem}-{strategy}.json").write_text(json.dumps(traces[0], indent=2), encoding="utf-8")
            rows.append({"image": path.name, "strategy": strategy, **c, "detector_calls": traces[0]["detector_calls"],
                         "median_ms": statistics.median(t["timing_ms"]["total"] for t in traces)})
        print(f"{image_idx+1}/{len(images)} {path.name}", flush=True)
    with (args.output/"per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = {}
    for s in strategies:
        own = [r for r in rows if r["strategy"] == s]
        aggregate = {k: sum(r[k] for r in own) for k in ["tp", "fp", "fn", "small_tp", "small_gt"]}
        summaries[s] = {**summarize(aggregate), "median_ms": statistics.median(timings[s]),
                        "p95_ms": float(np.percentile(timings[s], 95)),
                        "mean_calls": statistics.mean(r["detector_calls"] for r in own)}
    # Paired bootstrap resamples images, retaining numerator/denominator coupling.
    by_key = {(r["image"], r["strategy"]): r for r in rows}
    rng = np.random.default_rng(args.seed)
    deltas = []
    for _ in range(2000):
        sampled = rng.integers(0, len(images), len(images))
        small_gt = sum(by_key[(images[i].name, "global")]["small_gt"] for i in sampled)
        if small_gt:
            deltas.append(sum(by_key[(images[i].name, "adaptive")]["small_tp"]-
                              by_key[(images[i].name, "uniform")]["small_tp"] for i in sampled)/small_gt)
    weights_path = Path(args.weights)
    report = {"schema_version": 1, "date": datetime.now(timezone(timedelta(hours=8))).date().isoformat(), "sample_size": len(images), "repeats": args.repeats,
              "seed": args.seed, "budget_extra_crops": args.budget, "imgsz": args.imgsz,
              "weights": weights_path.name, "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest() if weights_path.exists() else None,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "ultralytics": importlib.metadata.version("ultralytics"),
                              "numpy": np.__version__, "device": args.device,
                              "hardware": torch.cuda.get_device_name(0) if str(args.device) != "cpu" else platform.processor()},
              "config": {"proposal_conf": 0.08, "output_conf": 0.25, "match_iou": 0.5,
                         "merge_iou": 0.5, "tile_fraction": 0.5, "overlap": 0.25,
                         "boundary_margin": args.boundary_margin},
              "protocol": "micro P/R at fixed confidence and IoU; original-pixel small area <1024; no COCO AP or crowd-ignore handling",
              "limitations": ["Deterministic small validation subset, not representative of full COCO",
                              "Pretrained YOLOv8 weights; no new detector training",
                              "Tile count is a budget proxy; latency also includes content-dependent overhead",
                              "Timing repeats are diagnostic; repeat across devices for performance claims",
                              "Heuristic confidence/texture scores are not calibrated uncertainty"],
              "summary": summaries, "adaptive_minus_uniform_small_recall_ci95": np.percentile(deltas, [2.5,97.5]).tolist() if deltas else None,
              "images": records}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    def pct(v):
        return "N/A" if v is None else f"{v*100:.2f}%"
    md = ["# Real detector diagnostic run", "", f"Date: {report['date']}. {len(images)} COCO val2017 images, ordered by image ID. {args.repeats} repeats. Seed {args.seed}.",
          "", f"Internal crop-edge filtering margin: {args.boundary_margin}. This is fixed-threshold detection analysis, **not COCO mAP**. Small means original bbox area < 32² pixels. No crowd-ignore metadata is available in YOLO labels.", "",
          "| Strategy | Calls/image | Precision50 | Recall50 | Small recall50 | TP / FP / FN | Median ms |", "|---|---:|---:|---:|---:|---|---:|"]
    for s,c in summaries.items():
        md.append(f"| {s} | {c['mean_calls']:.1f} | {pct(c['precision50'])} | {pct(c['recall50'])} | {pct(c['small_recall50'])} | {c['tp']} / {c['fp']} / {c['fn']} | {c['median_ms']:.1f} |")
    md += ["", "## Reading the evidence", "", "Compare adaptive against uniform/random/evidence/texture at equal calls. Full is an exhaustive slicing reference with a larger budget. Inspect per-image CSV and trace JSON before claiming benefits.",
           "", f"Paired image-bootstrap 95% CI for adaptive minus uniform small recall: {report['adaptive_minus_uniform_small_recall_ci95']} (fractions, 2,000 resamples).",
           "", "## Limits", "", *[f"- {s}" for s in report["limitations"]], "", "Exact weights, dependencies, input hashes, and configuration are in summary.json."]
    (args.output/"RESULTS.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
