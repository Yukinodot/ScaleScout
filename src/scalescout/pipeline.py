from dataclasses import asdict, dataclass
from time import perf_counter
from .core import make_tiles, score_tiles, select_tiles, remap, nms, serialize_detection, filter_internal_edges


@dataclass(frozen=True)
class Config:
    strategy: str = "adaptive"
    budget: int = 3
    tile_fraction: float = 0.5
    overlap: float = 0.25
    proposal_conf: float = 0.08
    output_conf: float = 0.25
    merge_iou: float = 0.5
    exploration: float = 0.25
    diversity: float = 0.35
    seed: int = 0
    boundary_margin: float = 0.02

    def __post_init__(self):
        if not 0 <= self.proposal_conf <= self.output_conf <= 1:
            raise ValueError("Require 0 <= proposal_conf <= output_conf <= 1")
        if not isinstance(self.budget, int) or self.budget < 0:
            raise ValueError("Budget must be a nonnegative integer")
        if not 0 <= self.boundary_margin < 0.5:
            raise ValueError("Boundary margin must be in [0,0.5)")


def run(image, detector, config=Config()):
    start = perf_counter()
    # Proposal threshold is deliberately low. Only final outputs use output_conf.
    global_detections = detector.predict(image, config.proposal_conf)
    after_global = perf_counter()
    tiles = score_tiles(image, make_tiles(*image.size, config.tile_fraction, config.overlap), global_detections)
    chosen = select_tiles(tiles, config.budget, config.strategy, config.exploration, config.diversity, config.seed)
    after_selection = perf_counter()
    combined = list(global_detections)
    rejected_edges = []
    for tile in chosen:
        local = detector.predict(image.crop(tile.box), config.proposal_conf)
        kept, rejected = filter_internal_edges(local, tile, *image.size, config.boundary_margin)
        combined.extend(remap(kept, tile, *image.size))
        rejected_edges.extend(remap(rejected, tile, *image.size))
    after_crops = perf_counter()
    final = nms([d for d in combined if d.score >= config.output_conf], config.merge_iou)
    end = perf_counter()
    trace = {
        "config": asdict(config), "width": image.width, "height": image.height,
        "detector_calls": 1+len(chosen), "candidate_tiles": len(tiles),
        "tiles": [{**asdict(t), "selected": t in chosen, "order": chosen.index(t) if t in chosen else None} for t in tiles],
        "global_proposals": [serialize_detection(d) for d in global_detections],
        "detections": [serialize_detection(d) for d in final],
        "rejected_internal_edges": [serialize_detection(d) for d in rejected_edges],
        "timing_ms": {"global": (after_global-start)*1000, "selection": (after_selection-after_global)*1000,
                      "crops": (after_crops-after_selection)*1000, "merge": (end-after_crops)*1000,
                      "total": (end-start)*1000},
    }
    return final, trace
