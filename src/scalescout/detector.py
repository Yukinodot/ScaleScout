from .core import Detection


class YOLODetector:
    def __init__(self, weights="yolov8n.pt", imgsz=640, device="cpu"):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz, self.device = imgsz, device

    def predict(self, image, conf):
        result = self.model.predict(image, imgsz=self.imgsz, device=self.device, conf=conf,
                                    iou=0.7, max_det=300, verbose=False)[0]
        return [Detection(tuple(map(float, b)), float(s), int(c)) for b, s, c in
                zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]

    def synchronize(self):
        import torch
        if str(self.device) != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def warmup(self):
        from PIL import Image
        self.predict(Image.new("RGB", (self.imgsz, self.imgsz)), 0.08)
        self.synchronize()
