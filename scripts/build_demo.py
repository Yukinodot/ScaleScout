"""Embed measured traces into an offline viewer. Includes a gain case and a failure case."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--report",type=Path,default=Path("reports/refined"))
p.add_argument("--output",type=Path,default=Path("demo/preview-data.js"))
args=p.parse_args()
summary=json.loads((args.report/"summary.json").read_text(encoding="utf-8"))
with (args.report/"per_image.csv").open(encoding="utf-8",newline="") as f:
    rows=list(csv.DictReader(f))
by_key={(r["image"],r["strategy"]):r for r in rows}
images=sorted({r["image"] for r in rows})
def gain(image):
    return int(by_key[image,"adaptive"]["small_tp"])-int(by_key[image,"uniform"]["small_tp"])
best=max(images,key=lambda x:(gain(x),int(by_key[x,"adaptive"]["tp"])-int(by_key[x,"uniform"]["tp"])))
worst=min(images,key=lambda x:(gain(x),int(by_key[x,"adaptive"]["tp"])-int(by_key[x,"uniform"]["tp"])))
examples=list(dict.fromkeys([best,worst,images[0]]))
names="person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush".split(",")
cases=[]
for image in examples:
    stem=Path(image).stem
    descriptions={best:"相对均匀策略的小目标收益案例",worst:"相对均匀策略的失败案例"}
    traces={s:json.loads((args.report/f"{stem}-{s}.json").read_text(encoding="utf-8")) for s in summary["summary"]}
    cases.append({"image":image,"label":f"{stem} · {descriptions.get(image,'常规案例')}","traces":traces})
payload={"summary":summary,"cases":cases,"names":names}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text("window.SCALESCOUT_DATA = "+json.dumps(payload,ensure_ascii=False,separators=(",",":"))+";\n",encoding="utf-8")
print(f"Built {len(cases)} real cases: {examples}")
