import numpy as np
import pytest
from PIL import Image
from scalescout.core import Detection, Tile, make_tiles, score_tiles, select_tiles, remap, nms, iou, filter_internal_edges
from scalescout.pipeline import Config, run
from scalescout.metrics import counts, read_yolo_labels, summarize


def test_tile_coverage_and_bounds():
    for w,h in [(317,223),(1,1),(31,2),(640,480)]:
        tiles = make_tiles(w,h)
        covered = np.zeros((h,w), dtype=bool)
        for t in tiles:
            x1,y1,x2,y2 = t.box
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            covered[y1:y2,x1:x2] = True
        assert covered.all()


@pytest.mark.parametrize("strategy", ["uniform","random","adaptive","texture","evidence"])
def test_budget_and_determinism(strategy):
    tiles = make_tiles(640,480)
    a = select_tiles(tiles,3,strategy,seed=99)
    assert len(a) == 3 and len({t.index for t in a}) == 3
    assert a == select_tiles(tiles,3,strategy,seed=99)
    assert select_tiles(tiles,0,strategy) == []
    assert len(select_tiles(tiles,100,strategy)) == len(tiles)


def test_no_proposals_still_explores_detail():
    image = np.zeros((100,100),dtype=np.uint8)
    image[60:,60:] = (np.indices((40,40)).sum(axis=0)%2)*255
    tiles = score_tiles(Image.fromarray(image), make_tiles(100,100), [])
    picked = select_tiles(tiles,1,"adaptive")
    assert picked[0].texture > 0
    assert picked[0].box[2:] == (100,100)


def test_crop_to_original_coordinates_and_clipping():
    d = Detection((0.,1.,30.,60.),.8,1)
    mapped = remap([d],Tile(2,(80,50,100,100)),100,100)[0]
    assert mapped.box == (80,51,100,100) and mapped.source == "tile:2"


def test_nms_duplicate_and_class_separation():
    a = Detection((0,0,10,10),.9,0)
    b = Detection((1,1,11,11),.8,0)
    c = Detection((1,1,11,11),.7,1)
    assert nms([b,c,a]) == [a,c]
    assert iou(a.box,a.box) == 1


def test_matching_no_double_count_and_wrong_class():
    gt = [Detection((0,0,10,10),1,0)]
    predictions = [Detection((0,0,10,10),.9,0),Detection((0,0,10,10),.8,0),Detection((0,0,10,10),.7,1)]
    assert counts(predictions,gt) == {"tp":1,"fp":2,"fn":0,"small_tp":1,"small_gt":1}


def test_wrong_class_leaves_false_negative():
    assert counts([Detection((0,0,10,10),.9,1)],[Detection((0,0,10,10),1,0)])["fn"] == 1


def test_empty_metrics_are_undefined():
    assert summarize(counts([],[]))["small_recall50"] is None
    assert summarize(counts([],[]))["precision50"] is None


def test_original_size_small_boundary():
    labels = [Detection((0,0,32,32),1,0),Detection((0,0,31,32),1,0)]
    assert counts([],labels)["small_gt"] == 1


def test_missing_label_is_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_yolo_labels(tmp_path/"missing.txt",100,100)


def test_label_pixel_conversion(tmp_path):
    path = tmp_path/"labels.txt"
    path.write_text("2 0.5 0.5 0.2 0.4\n")
    assert read_yolo_labels(path,100,50)[0].box == pytest.approx((40,15,60,35))


def test_budget_enforced_by_pipeline():
    class Fake:
        calls = 0
        def predict(self,image,conf):
            self.calls += 1
            return []
    detector = Fake()
    _, trace = run(Image.new("RGB",(100,100)),detector,Config(budget=2))
    assert detector.calls == trace["detector_calls"] == 3
    assert sum(t["selected"] for t in trace["tiles"]) == 2


@pytest.mark.parametrize("bad", [float("nan"),float("inf")])
def test_nonfinite_rejected(bad):
    with pytest.raises(ValueError):
        Detection((0,0,bad,10),.5,0)


def test_invalid_config():
    with pytest.raises(ValueError):
        Config(proposal_conf=.5,output_conf=.2)
    with pytest.raises(ValueError):
        select_tiles(make_tiles(10,10),-1)


def test_full_is_exhaustive():
    tiles=make_tiles(100,100)
    assert len(select_tiles(tiles,1,"full")) == len(tiles)


def test_internal_edge_filter_retains_original_boundary():
    border = Detection((0,10,20,30),.8,0)
    inside = Detection((5,10,20,30),.8,0)
    tile = Tile(0,(50,0,100,50))
    kept,rejected = filter_internal_edges([border,inside],tile,100,100)
    assert kept == [inside] and rejected == [border]
    assert filter_internal_edges([border],Tile(1,(0,0,50,50)),100,100)[0] == [border]
    assert filter_internal_edges([border],tile,100,100,margin=0)[0] == [border]


def test_invalid_edge_margin():
    with pytest.raises(ValueError):
        Config(boundary_margin=.5)
