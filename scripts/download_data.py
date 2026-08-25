#!/usr/bin/env python
"""
Download the NHANES transport files needed by this repository.

The survey files are public but are not redistributed here: they are large,
they are versioned by NCHS, and a reviewer should be able to see that the
analysis reads the official files rather than a copy we prepared.

    python scripts/download_data.py            # both cycles
    python scripts/download_data.py --cycle I  # one cycle

If your environment has no network access, download the URLs printed by
`--list` by hand and drop the files into data/.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

BASE = {
    "I": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/",
    "L": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/",
}
FILES = {
    "I": ["DEMO_I", "BIOPRO_I", "CUSEZN_I", "GLU_I", "INS_I", "DIQ_I",
          "DR1TOT_I", "DR2TOT_I", "BMX_I"],
    "L": ["DEMO_L", "BIOPRO_L", "GLU_L", "INS_L", "DIQ_L",
          "DR1TOT_L", "DR2TOT_L", "BMX_L"],
}


def url_for(cycle: str, stem: str) -> str:
    return f"{BASE[cycle]}{stem}.xpt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(cycles, list_only=False):
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cyc in cycles:
        for stem in FILES[cyc]:
            url = url_for(cyc, stem)
            dest = DATA / f"{stem}.xpt"
            if list_only:
                print(url)
                continue
            if dest.exists():
                print(f"  have {dest.name}")
            else:
                print(f"  get  {dest.name} <- {url}")
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as exc:                      # pragma: no cover
                    print(f"       FAILED: {exc}", file=sys.stderr)
                    continue
            manifest.append((dest.name, dest.stat().st_size, sha256(dest)))
    if manifest:
        out = DATA / "MANIFEST.sha256"
        with open(out, "w") as f:
            for name, size, h in manifest:
                f.write(f"{h}  {size:>12d}  {name}\n")
        print(f"\nchecksums written to {out}")
        print("Commit this file: it lets a reviewer confirm they analysed the "
              "same bytes we did.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", choices=["I", "L"], default=None)
    ap.add_argument("--list", action="store_true", dest="list_only")
    a = ap.parse_args()
    main([a.cycle] if a.cycle else ["I", "L"], list_only=a.list_only)
