"""Reconstruction from the presentation, not the group's original source or run."""
import argparse
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", default="coco.yaml")
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--device", default="0")
p.add_argument("--batch", type=int, default=32)
p.add_argument("--project", default="runs/baseline")
args = p.parse_args()
YOLO("yolov8n.yaml").train(data=args.data, epochs=args.epochs, imgsz=640, batch=args.batch,
    device=args.device, project=args.project, pretrained=False, seed=0,
    amp=True, lr0=0.001, box=7.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, flipud=0.0)
# auto_augment/erasing/dropout from the slide are classification-oriented options.
# They are deliberately not presented as verified YOLO detection augmentations.
