"""Detector-independent geometry and region scheduling. No labels enter selection."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    class_id: int
    source: str = "global"

    def __post_init__(self):
        x1, y1, x2, y2 = self.box
        if not all(math.isfinite(x) for x in (*self.box, self.score)):
            raise ValueError("Detection coordinates and score must be finite")
        if x2 <= x1 or y2 <= y1 or not 0 <= self.score <= 1:
            raise ValueError("Invalid detection geometry or confidence")


@dataclass(frozen=True)
class Tile:
    index: int
    box: tuple[int, int, int, int]
    texture: float = 0.0
    evidence: float = 0.0


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def nms(detections, threshold=0.5):
    """Class-aware greedy NMS, deterministic on ties."""
    if not 0 <= threshold <= 1:
        raise ValueError("NMS IoU must be in [0,1]")
    pending = sorted(detections, key=lambda d: (-d.score, d.class_id, d.box, d.source))
    kept = []
    while pending:
        best = pending.pop(0)
        kept.append(best)
        pending = [d for d in pending if d.class_id != best.class_id or iou(d.box, best.box) <= threshold]
    return kept


def _starts(length, size, overlap):
    if size >= length:
        return [0]
    step = max(1, round(size*(1-overlap)))
    return sorted(set([*range(0, length-size+1, step), length-size]))


def make_tiles(width, height, fraction=0.5, overlap=0.25):
    if width < 1 or height < 1 or not 0 < fraction <= 1 or not 0 <= overlap < 1:
        raise ValueError("Invalid image size, tile fraction, or overlap")
    tw, th = max(1, round(width*fraction)), max(1, round(height*fraction))
    boxes = [(x, y, x+tw, y+th) for y in _starts(height, th, overlap) for x in _starts(width, tw, overlap)]
    return [Tile(i, box) for i, box in enumerate(boxes)]


def score_tiles(image: Image.Image, tiles, proposals):
    """Texture finds proposal-free detail; confidence is a heuristic, not uncertainty calibration."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)/255.0
    result = []
    for tile in tiles:
        x1, y1, x2, y2 = tile.box
        crop = gray[y1:y2, x1:x2]
        dx = np.abs(np.diff(crop, axis=1)).mean() if crop.shape[1] > 1 else 0.0
        dy = np.abs(np.diff(crop, axis=0)).mean() if crop.shape[0] > 1 else 0.0
        evidence = 0.0
        for d in proposals:
            bx1, by1, bx2, by2 = d.box
            cx, cy = (bx1+bx2)/2, (by1+by2)/2
            if x1 <= cx < x2 and y1 <= cy < y2:
                relative_size = math.sqrt((bx2-bx1)*(by2-by1)/(image.width*image.height))
                smallness = max(0.0, 1-relative_size/0.15)
                evidence += smallness*(1-d.score)
        result.append(Tile(tile.index, tile.box, float((dx+dy)/2), evidence))
    return result


def select_tiles(tiles, budget, strategy="adaptive", exploration=0.25, diversity=0.35, seed=0):
    """Adaptive selection reserves texture-led coverage to avoid dependence on existing detections.

    Uniform uses farthest-center coverage; random is a seeded equal-budget baseline.
    All strategies use the same candidate crops. Full inspects all candidates.
    """
    if not isinstance(budget, int) or budget < 0 or not 0 <= exploration <= 1 or diversity < 0:
        raise ValueError("Invalid selection parameters")
    if strategy not in {"global", "uniform", "random", "adaptive", "evidence", "texture", "full"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    if strategy == "global" or not tiles or budget == 0:
        return []
    if strategy == "full":
        return list(tiles)
    k = min(budget, len(tiles))
    if strategy == "random":
        indices = np.random.default_rng(seed).choice(len(tiles), k, replace=False)
        return [tiles[int(i)] for i in indices]
    max_t = max(t.texture for t in tiles) or 1
    max_e = max(t.evidence for t in tiles) or 1
    selected, remaining = [], list(tiles)
    reserve = math.ceil(k*exploration) if strategy == "adaptive" else 0
    for step in range(k):
        def utility(t):
            redundancy = max((iou(t.box, s.box) for s in selected), default=0)
            texture, evidence = t.texture/max_t, t.evidence/max_e
            if strategy == "uniform":
                center = ((t.box[0]+t.box[2])/2, (t.box[1]+t.box[3])/2)
                if not selected:
                    return -t.index
                return min((center[0]-(s.box[0]+s.box[2])/2)**2 + (center[1]-(s.box[1]+s.box[3])/2)**2 for s in selected)
            if strategy == "texture" or (strategy == "adaptive" and step < reserve):
                score = texture
            elif strategy == "evidence":
                score = evidence
            else:
                score = 0.7*evidence + 0.3*texture
            return score-diversity*redundancy
        best = max(remaining, key=lambda t: (utility(t), -t.index))
        selected.append(best)
        remaining.remove(best)
    return selected


def remap(detections, tile, width, height):
    """Crop predictions are already in crop pixels, not resized network pixels."""
    x, y, _, _ = tile.box
    mapped = []
    for d in detections:
        b = (max(0, min(width, d.box[0]+x)), max(0, min(height, d.box[1]+y)),
             max(0, min(width, d.box[2]+x)), max(0, min(height, d.box[3]+y)))
        if b[2] > b[0] and b[3] > b[1]:
            mapped.append(Detection(b, d.score, d.class_id, f"tile:{tile.index}"))
    return mapped


def filter_internal_edges(detections, tile, width, height, margin=0.02):
    """Reject crop fragments near internal boundaries; retain actual image-edge objects.

    The margin is relative to crop size. Zero disables the filter. This may also
    remove true objects and is an exploratory precision/recall tradeoff.
    """
    if not 0 <= margin < 0.5:
        raise ValueError("Boundary margin must be in [0,0.5)")
    if margin == 0:
        return list(detections), []
    x1,y1,x2,y2 = tile.box
    tw,th = x2-x1,y2-y1
    kept,rejected = [],[]
    for d in detections:
        a,b,c,e = d.box
        clipped = ((x1 > 0 and a <= tw*margin) or (y1 > 0 and b <= th*margin) or
                   (x2 < width and c >= tw*(1-margin)) or (y2 < height and e >= th*(1-margin)))
        (rejected if clipped else kept).append(d)
    return kept,rejected


def serialize_detection(d):
    return asdict(d)
