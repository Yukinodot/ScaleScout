import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from .pipeline import Config, run


def load_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Explainable budgeted YOLO zoom")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--budget", type=int, default=3, help="Extra crops, excluding the global call")
    parser.add_argument("--strategy", choices=["global","uniform","random","adaptive","evidence","texture","full"], default="adaptive")
    parser.add_argument("--tile-fraction", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--boundary-margin", type=float, default=0.02, help="Crop-edge filter; 0 disables")
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction"))
    args = parser.parse_args()
    from .detector import YOLODetector
    image = load_image(args.image)
    detector = YOLODetector(args.weights, args.imgsz, args.device)
    detector.warmup()
    detections, trace = run(image, detector, Config(strategy=args.strategy, budget=args.budget,
                                                  tile_fraction=args.tile_fraction, output_conf=args.conf,
                                                  boundary_margin=args.boundary_margin))
    trace["input_sha256"] = hashlib.sha256(args.image.read_bytes()).hexdigest()
    trace["weights"] = args.weights
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    draw = ImageDraw.Draw(image)
    for d in detections:
        draw.rectangle(d.box, outline="#00daaf", width=2)
        draw.text((d.box[0], d.box[1]), f"{d.class_id} {d.score:.2f}", fill="#00daaf")
    image.save(args.output/"detections.jpg")
    print(f"{len(detections)} detections, {trace['detector_calls']} calls. Trace: {args.output/'trace.json'}")


if __name__ == "__main__":
    main()
