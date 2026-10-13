"""Download a deterministic COCO val2017 subset. Never commits images or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    part = path.with_suffix(path.suffix+".part")
    urllib.request.urlretrieve(url, part)
    part.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--subset-name", default="coco-mini")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    if args.offset < 0 or not args.subset_name.replace("-", "").isalnum():
        parser.error("offset must be nonnegative and subset-name must be alphanumeric/hyphen")
    subset = args.root/args.subset_name
    archive = args.root/"coco2017labels.zip"
    download(LABELS_URL, archive)
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if "/labels/val2017/" in n and n.endswith(".txt"))[args.offset:args.offset+args.limit]
        if len(names) != args.limit:
            raise RuntimeError("Insufficient validation label files")
        records = []
        for name in names:
            # Rebuild paths from a numeric image identifier instead of trusting archive paths.
            stem = Path(name).stem
            if len(stem) != 12 or not stem.isdecimal():
                raise ValueError(f"Unexpected image id: {stem}")
            label = subset/"labels"/f"{stem}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_bytes(z.read(name))
            records.append({"id": stem, "url": f"http://images.cocodataset.org/val2017/{stem}.jpg"})
    def get_image(record):
        path = subset/"images"/(record["id"]+".jpg")
        download(record["url"], path)
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return record
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(get_image, records))
    manifest = {"split": "COCO val2017", "sampling": "N labeled images by numeric ID from offset; diagnostic, not representative",
                "offset":args.offset,"limit":args.limit,
                "labels_url": LABELS_URL, "labels_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "crowd_policy": "Ultralytics detection labels; crowd/ignore metadata unavailable", "images": records}
    (subset/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded {len(records)} labeled validation images to {subset}")


if __name__ == "__main__":
    main()
