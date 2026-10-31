"""Native23 compact cache: bounded-memory conversion, strict metadata, atomic publish.

Only representation checks reject input here. Appendix E selection belongs to the
manifest; collision/velocity/limit diagnostics do not remove additional motions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from supermimic.data.motion_convert import g1_xml_path
from supermimic.data.quality import validate_reference
from supermimic.robots.spec import JOINT_NAMES, STATE_DIM

SCHEMA = "sm23-aligned-v2"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_names(path):
    with Path(path).open(newline="") as f:
        names = [r["name"] for r in csv.DictReader(f) if r.get("keep", "1") in ("1", "true", "True")]
    if not names or len(names) != len(set(names)):
        raise ValueError("Manifest must contain nonempty, unique kept motion names")
    return sorted(names)


def validate_pack(pack, manifest=None):
    if pack.get("schema") not in ("sm23-aligned-v1", SCHEMA):
        raise ValueError("Expected a native23 cache built by sm23 pack")
    ref = pack["ref"]
    names, starts, lengths = pack["names"], pack["starts"], pack["lengths"]
    if not names or len(names) != len(set(names)) or not (len(names) == len(starts) == len(lengths)):
        raise ValueError("Invalid packed clip table")
    end = 0
    for start, length in zip(starts, lengths):
        if int(start) != end or int(length) < 2:
            raise ValueError("Packed clips must be contiguous and have at least two frames")
        end += int(length)
    if ref.shape != (end, STATE_DIM) or ref.dtype not in (torch.float16, torch.float32):
        raise ValueError("Invalid packed native23 reference tensor")
    if pack["schema"] == SCHEMA:
        if pack["fps"] != 50 or tuple(pack["joint_names"]) != JOINT_NAMES:
            raise ValueError("Packed fps/joint order differs from native23 50Hz contract")
        if pack["model_sha256"] != sha256(g1_xml_path()):
            raise ValueError("Packed model hash differs from current native23 model")
        if pack["root_angular_velocity_frame"] != "world" or pack["quaternion_order"] != "wxyz":
            raise ValueError("Packed reference coordinate convention mismatch")
    if manifest is not None and sorted(names) != manifest_names(manifest):
        raise ValueError("Packed motions differ from the selected manifest")


def load_pack(path, manifest=None):
    pack = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validate_pack(pack, manifest)
    return pack


def _frame_count(path):
    # Read only the NPY header, not the compressed body arrays, for allocation.
    with zipfile.ZipFile(path) as z, z.open("joint_pos.npy") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported NPY header {version}: {path}")
    if len(shape) != 2 or shape[1] != len(JOINT_NAMES) or shape[0] < 2:
        raise ValueError(f"Invalid joint shape in {path}: {shape}")
    return shape[0]


def _read_reference(path, model_hash):
    with np.load(path, allow_pickle=False) as z:
        # Materialize once; repeated NpzFile indexing otherwise decompresses repeatedly.
        d = {k: z[k] for k in ("fps", "timestamps", "joint_names", "body_names", "joint_pos",
             "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "metadata_json")}
    validate_reference(d)
    if json.loads(d["metadata_json"].item()).get("model_sha256") != model_hash:
        raise ValueError(f"Reference model hash mismatch: {path}")
    root = d["body_names"].tolist().index("pelvis")
    ref = np.concatenate([d["body_pos_w"][:, root], d["body_quat_w"][:, root], d["joint_pos"],
        d["body_lin_vel_w"][:, root], d["body_ang_vel_w"][:, root], d["joint_vel"]], axis=1)
    with np.errstate(over="ignore"):
        compact = ref.astype(np.float16)
    if not np.isfinite(compact).all():
        raise ValueError(f"fp16 overflow in {path}; do not silently clip reference values")
    return compact, np.abs(compact.astype(np.float32) - ref).max(axis=0)


def build_library(source, output, manifest=None, workers=8):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Cache already exists: {output}; use a new output path")
    if workers < 1:
        raise ValueError("workers must be positive")
    names = manifest_names(manifest) if manifest else sorted(p.stem for p in source.glob("*.npz"))
    if not names:
        raise ValueError("No reference clips")
    files = [source / (n + ".npz") for n in names]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_hash = sha256(g1_xml_path())
    start_time = time.monotonic()
    print(f"Indexing {len(files)} native23 references", flush=True)
    lengths = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded submissions, unlike executor.map over 130k large results.
        for offset in range(0, len(files), workers * 8):
            lengths.extend(pool.map(_frame_count, files[offset:offset + workers * 8]))
    starts = (np.cumsum(lengths, dtype=np.int64) - lengths).tolist()
    frames = sum(lengths)
    errors = np.zeros(STATE_DIM)
    print(f"Packing {frames:,} frames into {frames * STATE_DIM * 2 / 2**30:.3f} GiB fp16", flush=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".building-", dir=output.parent) as work:
        data = np.memmap(Path(work) / "ref.bin", mode="w+", dtype=np.float16, shape=(frames, STATE_DIM))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(files), workers * 4):
                batch = files[offset:offset + workers * 4]
                for i, (ref, err) in enumerate(pool.map(lambda p: _read_reference(p, model_hash), batch), offset):
                    if len(ref) != lengths[i]:
                        raise ValueError(f"Reference changed during packing: {files[i]}")
                    data[starts[i]:starts[i] + lengths[i]] = ref
                    errors = np.maximum(errors, err)
                done = offset + len(batch)
                if offset == 0 or done == len(files) or done // 1024 != offset // 1024:
                    print(f"Validated/packed {done}/{len(files)} ({time.monotonic()-start_time:.1f}s)", flush=True)
        data.flush()
        pack = dict(schema=SCHEMA, fps=50, joint_names=list(JOINT_NAMES), model_sha256=model_hash,
            quaternion_order="wxyz", root_angular_velocity_frame="world", ref=torch.from_numpy(data),
            names=names, starts=starts, lengths=lengths,
            manifest_sha256=sha256(manifest) if manifest else None,
            max_abs_fp16_error_per_column=errors.tolist())
        validate_pack(pack, manifest)
        pending = Path(work) / "library.pt"
        torch.save(pack, pending)
        validate_pack(load_pack(pending), manifest)
        pending.replace(output)
    report = dict(schema=SCHEMA, output=str(output), clips=len(names), frames=frames, fps=50,
        duration_hours=(frames-len(names))/50/3600, tensor_gib=frames*STATE_DIM*2/2**30,
        file_bytes=output.stat().st_size, model_sha256=model_hash,
        manifest_sha256=pack["manifest_sha256"], max_abs_fp16_error_per_column=errors.tolist(),
        elapsed_s=time.monotonic()-start_time, validation="all input arrays, timestamps, joints, model hash, finite fp16",
        selection="manifest only; additional native23 quality checks remain diagnostic")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
