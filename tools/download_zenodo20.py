"""Download the official COVID-19 CT Lung and Infection Segmentation Dataset.

The files are fetched from Zenodo record 3757476. Downloads resume from a
``.part`` file and are verified against the file sizes published by Zenodo.
"""

from __future__ import annotations

from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "archives" / "zenodo20"
RECORD_API = "https://zenodo.org/api/records/3757476"
REQUIRED_FILES = {
    "COVID-19-CT-Seg_20cases.zip",
    "Infection_Mask.zip",
    "Lung_Mask.zip",
    "Lung_and_Infection_Mask.zip",
}


def download_file(url: str, destination: Path, expected_size: int) -> None:
    if destination.is_file() and destination.stat().st_size == expected_size:
        print(f"Already present: {destination.name}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.is_file() and partial.stat().st_size == expected_size:
        partial.replace(destination)
        print(f"Recovered completed download: {destination.name}")
        return
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError(f"Partial download is larger than the official file: {partial}")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    append = offset > 0 and response.status_code == 206
    if offset and not append:
        offset = 0
    mode = "ab" if append else "wb"
    with partial.open(mode) as output:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                output.write(chunk)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Incomplete download for {destination.name}: {partial.stat().st_size} of {expected_size} bytes"
        )
    partial.replace(destination)
    print(f"Downloaded: {destination.name}")


def main() -> None:
    record_response = requests.get(RECORD_API, timeout=30)
    record_response.raise_for_status()
    record = record_response.json()
    files = {item["key"]: item for item in record["files"]}
    missing = REQUIRED_FILES - files.keys()
    if missing:
        raise RuntimeError(f"Zenodo record no longer exposes: {sorted(missing)}")
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    for filename in sorted(REQUIRED_FILES):
        item = files[filename]
        download_file(item["links"]["self"], ARCHIVES / filename, item["size"])
    print("All Zenodo20 source archives are available. Run: python tools/prepare_zenodo20.py")


if __name__ == "__main__":
    main()
