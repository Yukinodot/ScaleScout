"""Fixed-threshold diagnostics, not a replacement for official COCO AP."""
from .core import Detection, iou


def match(predictions, labels, threshold=0.5):
    """Score-ordered class-aware one-to-one matching."""
    used, pairs, fp = set(), [], []
    for pred in sorted(predictions, key=lambda d: -d.score):
        candidates = [(iou(pred.box, gt.box), i) for i, gt in enumerate(labels)
                      if i not in used and gt.class_id == pred.class_id]
        overlap, idx = max(candidates, default=(0.0, -1))
        if idx >= 0 and overlap >= threshold:
            used.add(idx)
            pairs.append((pred, idx))
        else:
            fp.append(pred)
    return pairs, fp, [i for i in range(len(labels)) if i not in used]


def counts(predictions, labels, threshold=0.5):
    pairs, false_positives, missed = match(predictions, labels, threshold)
    small_ids = {i for i, d in enumerate(labels) if (d.box[2]-d.box[0])*(d.box[3]-d.box[1]) < 32**2}
    return {"tp": len(pairs), "fp": len(false_positives), "fn": len(missed),
            "small_tp": sum(i in small_ids for _, i in pairs), "small_gt": len(small_ids)}


def summarize(c):
    divide = lambda a, b: a/b if b else None
    return {**c, "precision50": divide(c["tp"], c["tp"]+c["fp"]),
            "recall50": divide(c["tp"], c["tp"]+c["fn"]), "small_recall50": divide(c["small_tp"], c["small_gt"])}


def read_yolo_labels(path, width, height):
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 5:
            raise ValueError(f"Expected detection labels (5 columns): {path}")
        c, cx, cy, w, h = map(float, tokens)
        if c != int(c) or c < 0 or not all(0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid normalized label: {path}")
        labels.append(Detection(((cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height), 1., int(c), "label"))
    return labels
