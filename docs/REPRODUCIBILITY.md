# 运行记录与可复现范围

仓库报告由真实 YOLOv8n 推理生成，初始/优化两轮都使用同一批 64 张 COCO val2017 图片。之后冻结参数，检查按 ID 排序的后续 64 张图片，总共 128 张不同图片。没有运行完整 COCO 训练，也没有运行开放词汇示例。

## 本次命令

```bash
python -m pip install -e ".[yolo,dev]"
python -m pytest -q
python scripts/fetch_sample.py --root data --limit 64
python scripts/benchmark.py --images data/coco-mini/images --labels data/coco-mini/labels --device 0 --limit 64 --budget 3 --repeats 2 --boundary-margin 0 --output reports/initial
python scripts/benchmark.py --images data/coco-mini/images --labels data/coco-mini/labels --device 0 --limit 64 --budget 3 --repeats 2 --boundary-margin 0.02 --output reports/refined
python scripts/build_demo.py --report reports/refined
python scripts/fetch_sample.py --root data --offset 64 --limit 64 --subset-name coco-holdout
python scripts/benchmark.py --images data/coco-holdout/images --labels data/coco-holdout/labels --device 0 --limit 64 --budget 3 --repeats 2 --boundary-margin 0.02 --output reports/holdout
```

初始运行发生在边缘过滤实现前，等价于当前 `--boundary-margin 0`。参数 0.02 在查看初始结果后确定，优化结果属于同一数据上的探索性复测。后续 64 图与前 64 图没有重叠；该检查仍属于同一数据集的小样本诊断，尚无跨领域验证。演示面板展示第一批优化后的案例和统计，留出结果在独立报告中。

## 证据

- `summary.json`：设备、关键依赖版本、权重 SHA256、图片与标签 SHA256。
- `per_image.csv`：所有图片与策略的计数和耗时。
- 每图每策略的 `.json`：区域选择、全图候选、最终输出及各阶段耗时。各轮完整 448 份记录装在 `traces.zip`，`python scripts/unpack_traces.py` 校验并解压。
- `data-manifest.json`：下载来源、标签压缩包哈希和样本选择规则。
- `requirements-tested.txt`：本次关键运行版本，不强行约束用户设备的 CUDA 包。
- `VERIFICATION.md`：本地验证状态与未执行的入口。

初始记录在加入边缘过滤实现后补充了 `boundary_margin=0` 的重放元数据，预测结果没有修改。压缩脚本逐字节检查压缩前后内容一致，完整证据均保留。

数据有重叠标注和 crowd/ignore 等复杂情况，YOLO 格式没有完整评测元数据。当前计数用于调度诊断，正式 COCO 比较应使用官方标注及 pycocotools。

GPU/CPU、操作系统和库版本改变时，预测或耗时可能变化。两次重复只支持本机诊断，不支持宣称稳定实时性能。图片和权重不在仓库中，脚本会从公开来源获取；使用哈希检查是否与本次一致。
