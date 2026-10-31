#!/usr/bin/env python3
"""Download BONES-SEED with HF authentication and automatic partial-download reuse.

Run with ../SOMA-Retargeter/.venv/bin/python. Accept dataset access conditions
on Hugging Face, then run ../SOMA-Retargeter/.venv/bin/hf auth login locally.
No credentials are accepted as CLI arguments or written into the manifest.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import huggingface_hub
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

REPO = "bones-studio/seed"
UNIFORM_PATTERNS = ["soma_uniform.tar.gz", "metadata/*", "soma_shapes/soma_base_fit_mhr_params.npz",
                    "soma_shapes/soma_base_rig/*", "README.md", "LICENSE.md", ".gitattributes"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path,
                   default=Path(__file__).resolve().parents[1] / "data/raw/bones_seed")
    p.add_argument("--variant", choices=["uniform", "all"], default="uniform")
    p.add_argument("--extract", action="store_true", help="Also extract downloaded motion archives")
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    status_path = a.output / "download_status.json"
    status = {"repo": REPO, "variant": a.variant, "output": str(a.output),
              "updated_at": datetime.now(timezone.utc).isoformat(),
              "huggingface_hub_version": huggingface_hub.__version__}

    def save(state, **extra):
        status.update(state=state, **extra)
        status_path.write_text(json.dumps(status, indent=2))

    try:
        # Verify access before fetching or extracting anything from the gated dataset.
        meta = get_hf_file_metadata(hf_hub_url(REPO, "soma_uniform.tar.gz", repo_type="dataset"))
        revision = meta.commit_hash
        info = HfApi().dataset_info(REPO, revision=revision, files_metadata=True)
        from fnmatch import fnmatch
        selected = [f for f in info.siblings if a.variant == "all" or
                    any(fnmatch(f.rfilename, pattern) for pattern in UNIFORM_PATTERNS)]
        total_bytes = sum(f.size or 0 for f in selected)
        remaining = sum(max(0, (f.size or 0) - ((a.output / f.rfilename).stat().st_size
                        if (a.output / f.rfilename).is_file() else 0)) for f in selected)
        if shutil.disk_usage(a.output).free < remaining:
            raise RuntimeError("Insufficient space for selected downloads")
        save("downloading", revision=revision, total_bytes=total_bytes,
             selected_files=[{"path": f.rfilename, "bytes": f.size} for f in selected])
        print(f"Downloading {len(selected)} files ({total_bytes / 1e9:.2f} GB) to {a.output}", flush=True)
        snapshot_download(REPO, repo_type="dataset", revision=revision, local_dir=a.output,
                          allow_patterns=None if a.variant == "all" else UNIFORM_PATTERNS,
                          max_workers=4)
        for f in selected:
            path = a.output / f.rfilename
            if not path.is_file() or (f.size is not None and path.stat().st_size != f.size):
                raise RuntimeError(f"Downloaded file missing or size mismatch: {f.rfilename}")
        save("downloaded", size_validation_passed=True)
        if a.extract:
            import tarfile
            for f in selected:
                if f.rfilename.endswith(".tar.gz"):
                    save("extracting", extracting_archive=f.rfilename)
                    print(f"Extracting {f.rfilename}", flush=True)
                    with tarfile.open(a.output / f.rfilename, mode="r|gz") as archive:
                        archive.extractall(a.output, filter="data")
            save("complete", extracted=True)
        print(f"Download complete: {a.output}")
    except HfHubHTTPError as exc:
        code = exc.response.status_code
        if code in (401, 403):
            save("needs_huggingface_access", http_status=code)
            print("Access denied. First obtain access at https://huggingface.co/datasets/bones-studio/seed "
                  "and run ../SOMA-Retargeter/.venv/bin/hf auth login in your terminal. "
                  "Then rerun this command; partial downloads are reused.", file=sys.stderr)
            return 2
        save("http_error", http_status=code)
        print(f"Hugging Face returned HTTP {code}; rerun to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
