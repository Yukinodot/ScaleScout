"""Store per-image evidence in compact archives; originals remain on disk."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

for directory in sorted(Path("reports").iterdir()):
    if not directory.is_dir():
        continue
    files=sorted(p for p in directory.glob("*.json") if re.fullmatch(r"\d{12}-(global|uniform|random|adaptive|evidence|texture|full)\.json",p.name))
    if not files:
        continue
    archive=directory/"traces.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            # Fixed archive metadata makes repeated packaging byte-for-byte reproducible.
            entry=zipfile.ZipInfo(p.name,date_time=(2026,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(entry,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(files)
        assert all(z.read(p.name)==p.read_bytes() for p in files)
    receipt={"archive":"traces.zip","count":len(files),"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
             "contents":"Full per-image prediction and selection trace JSON; no images or weights"}
    (directory/"archive-manifest.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(directory,receipt["count"],archive.stat().st_size,"bytes")
