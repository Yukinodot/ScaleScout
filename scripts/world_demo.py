"""Optional continuation of the presentation's YOLO-World demo; not part of ScaleScout evaluation."""
import argparse
from ultralytics import YOLOWorld

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("--classes", nargs="+", default=["cup", "laptop", "mouse", "book"])
p.add_argument("--device", default="cpu")
args = p.parse_args()
model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(args.classes)
model.predict(args.image, device=args.device, save=True, project="outputs", name="world")
