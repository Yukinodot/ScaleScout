"""Restore archived traces safely and verify their archive hash."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

for directory in sorted(Path("reports").iterdir()):
    archive=directory/"traces.zip"
    if not archive.exists():
        continue
    manifest=json.loads((directory/"archive-manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest["sha256"]:
        raise ValueError(f"Archive hash mismatch: {directory}")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or len(z.namelist())!=manifest["count"]:
            raise ValueError(f"Invalid archive: {directory}")
        for name in z.namelist():
            if not re.fullmatch(r"\d{12}-(global|uniform|random|adaptive|evidence|texture|full)\.json",name):
                raise ValueError(f"Unexpected entry: {name}")
            target=directory/name
            data=z.read(name)
            if target.exists() and target.read_bytes()!=data:
                raise ValueError(f"Refusing to overwrite a different trace: {target}")
            target.write_bytes(data)
    print(f"Verified and restored {manifest['count']} traces in {directory}")
