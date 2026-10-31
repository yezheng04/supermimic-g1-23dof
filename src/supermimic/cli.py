"""Data preparation, training, deterministic evaluation and inspection entrypoints."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np


def apply_config(obj, values):
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown configuration key: {type(obj).__name__}.{key}")
        old = getattr(obj, key)
        if dataclasses.is_dataclass(old):
            apply_config(old, value)
        else:
            if isinstance(old, tuple):
                value = tuple(value)
            setattr(obj, key, value)
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("convert")
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--input-fps", type=float)
    p.add_argument("--output-fps", type=float, default=50.)
    p.add_argument("--project-29", action="store_true")
    p = subs.add_parser("demo")
    p.add_argument("--output", type=Path, default=Path("data/npz/demo/stand.npz"))
    p = subs.add_parser("train")
    p.add_argument("--config", type=Path, default=Path("configs/rtx5080.json"))
    p.add_argument("--data", type=Path)
    p.add_argument("--resume", type=Path)
    p = subs.add_parser("evaluate")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--device", default="cuda:0")
    p = subs.add_parser("inspect")
    p.add_argument("reference", type=Path)
    p = subs.add_parser("pack")
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p = subs.add_parser("play-reference")
    p.add_argument("reference", type=Path)
    args = parser.parse_args()
    if args.command == "convert":
        from supermimic.data.motion_convert import load_csv, map_joints, save_reference
        if args.source.suffix == ".csv":
            raw = load_csv(args.source, project_29=args.project_29)
            ts, fps = None, args.input_fps
        elif args.source.suffix == ".npz":
            with np.load(args.source, allow_pickle=False) as d:
                # Source-rate robot poses, not an already converted reference NPZ.
                order = str(d["quaternion_order"].item())
                q = d["root_quat"]
                if order == "wxyz":
                    q = q[:, [1,2,3,0]]
                elif order != "xyzw":
                    raise ValueError("Explicit quaternion_order must be xyzw or wxyz")
                raw = map_joints(np.c_[d["root_pos"], q, d["joint_pos"]], d["joint_names"], args.project_29)
                ts = d["timestamps"].copy() if "timestamps" in d else None
                fps = args.input_fps
                if "fps" in d:
                    source_fps = float(d["fps"].item())
                    if fps is not None and not np.isclose(fps, source_fps):
                        raise ValueError("--input-fps conflicts with raw NPZ metadata")
                    fps = source_fps
        else:
            raise ValueError("Use CSV or source-rate robot NPZ exported by scripts/retarget_soma.py")
        report = save_reference(raw, args.output, fps, args.output_fps, ts, args.source,
                                extra_metadata={"projected_from_29": args.project_29})
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "demo":
        from supermimic.data.motion_convert import save_reference
        from supermimic.data.synthetic import standing_raw
        print(json.dumps(save_reference(standing_raw(), args.output, 30., source="synthetic standing fixture; not a training dataset"), indent=2))
    elif args.command == "train":
        from supermimic.config import TrainConfig
        from supermimic.training.loop import Trainer
        cfg = apply_config(TrainConfig(), json.loads(args.config.read_text()))
        if args.data:
            if args.data.suffix == ".pt":
                cfg.data.library = args.data
            else:
                cfg.data.library = None
                cfg.data.npz_dir = args.data
        if args.resume:
            cfg.resume = args.resume
        if cfg.env.future_frames != cfg.policy_trainer.future_frames or cfg.policy.history != cfg.policy_trainer.history:
            raise ValueError("Environment, policy and imagination history/goal settings must agree")
        trainer = Trainer(cfg)
        try:
            trainer.run()
        finally:
            trainer.env.close()
    elif args.command == "evaluate":
        from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results, summarize
        from supermimic.envs.mdp.motion_library import discover_npz
        from supermimic.data.packed_library import load_pack
        policy, cfg = load_policy_from_checkpoint(args.checkpoint, args.device)
        if args.data.suffix == ".pt":
            pack = load_pack(args.data, args.manifest)
            del pack
            files = [str(args.data)]
        else:
            files = list(map(str, discover_npz(args.data, manifest=args.manifest)))
        if not files:
            raise ValueError("No reference clips")
        results = run_benchmark(policy, files, device=args.device, num_envs=args.num_envs,
            future_frames=cfg["env"]["future_frames"], history_length=cfg["policy"]["history"],
            anchor_obs=cfg["env"].get("anchor_obs", False), paper_protocol=cfg["env"].get("paper_protocol", True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_results(results, args.output)
        print(json.dumps(summarize(results), indent=2))
    elif args.command == "inspect":
        from supermimic.data.quality import validate_reference
        with np.load(args.reference, allow_pickle=False) as d:
            validate_reference(d)
            print(d["metadata_json"].item())
    elif args.command == "pack":
        from supermimic.data.packed_library import build_library
        print(json.dumps(build_library(args.source, args.output, args.manifest, args.workers), indent=2))
    else:
        from supermimic.data.playback import play_reference
        play_reference(args.reference)


if __name__ == "__main__":
    main()
