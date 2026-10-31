"""Live W&B logger for rank-zero JSONL/TensorBoard outputs, including backfill.

Runs independently of simulation: adding logging to an existing job requires no
trainer restart. Uses a custom iteration axis so late-flushed TB events are valid.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import time


def read_updates(path: Path, offset: int):
    """Return complete JSON lines only; an unfinished write remains for next poll."""
    if not path.exists():
        return [], offset
    if path.stat().st_size < offset:
        offset = 0
    rows = []
    with path.open("rb") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            values = {"iteration": int(row["iteration"])}
            for group in ("wm", "policy"):
                for key, value in row[group].items():
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite update metric: {group}/{key}")
                    values[f"{group}/{key}"] = value
            rows.append(values)
            offset = f.tell()
    return rows, offset


def tensorboard_rows(accumulator, cursors):
    """Use event walltime per tag, allowing checkpoint resumes with repeated steps."""
    accumulator.Reload()
    grouped = {}
    next_cursors = dict(cursors)
    for tag in accumulator.Tags().get("scalars", []):
        cutoff = cursors.get(tag, -1.)
        for event in accumulator.Scalars(tag):
            if event.wall_time <= cutoff:
                continue
            next_cursors[tag] = max(next_cursors.get(tag, -1.), event.wall_time)
            if not math.isfinite(event.value):
                continue  # Warmup placeholders are not numeric training measurements.
            key = f"periodic/{tag}" if tag.startswith(("wm/", "policy/")) else tag
            grouped.setdefault(event.step, {"iteration": event.step})[key] = event.value
    return [grouped[k] for k in sorted(grouped)], next_cursors


def save_state(path, state):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(state, indent=2) + "\n")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", default="supermimic-g1-23dof")
    parser.add_argument("--entity")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--poll-seconds", type=float, default=5.)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status-file", type=Path, help="Optional trainer status file; finish when COMPLETED or FAILED")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    folder = args.run_dir.resolve()
    config = json.loads((folder / "config.json").read_text())
    lock = (folder / "wandb_logger.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import wandb
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    state_file = folder / "wandb_logger_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if any(state.get(k) != getattr(args, k) for k in ("project", "entity", "mode")):
            raise ValueError("Existing W&B logger belongs to another project/entity/mode")
    else:
        state = dict(id=wandb.util.generate_id(), project=args.project, entity=args.entity,
            mode=args.mode, updates_offset=0, tb_walltime={}, latest_training_iteration=0)
        save_state(state_file, state)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    tb = EventAccumulator(str(folder / "tb"), size_guidance={"scalars": 0}, purge_orphaned_data=False)
    with wandb.init(project=args.project, entity=args.entity, id=state["id"],
        resume="allow" if args.mode == "online" else None, name=config["run_name"],
        dir=str(folder), mode=args.mode, config={**config, "logger_source": "rank0 JSONL and TensorBoard"},
        save_code=False, settings=wandb.Settings(console="off", disable_git=True)) as run:
        run.define_metric("iteration")
        run.define_metric("*", step_metric="iteration", step_sync=False)
        state["url"] = run.url
        state["status"] = "running"
        save_state(state_file, state)
        print(f"W&B run: {run.url or state['id']}", flush=True)
        while True:
            rows, offset = read_updates(folder / "updates.jsonl", state["updates_offset"])
            tb_rows, tb_cursors = tensorboard_rows(tb, state["tb_walltime"])
            for values in rows + tb_rows:
                run.log(values)
                state["latest_training_iteration"] = max(state["latest_training_iteration"], values["iteration"])
            state.update(updates_offset=offset, tb_walltime=tb_cursors, last_poll_unix=time.time())
            run.summary["latest_training_iteration"] = state["latest_training_iteration"]
            save_state(state_file, state)
            if rows or tb_rows:
                print(f"Logged {len(rows)} updates, {len(tb_rows)} TB rows; latest iteration {state['latest_training_iteration']}", flush=True)
            if args.status_file is not None and args.status_file.exists():
                training_status = args.status_file.read_text().strip()
                run.summary["training_status"] = training_status
                if training_status.startswith("FAILED"):
                    state["status"] = "training_failed"
                    save_state(state_file, state)
                    raise RuntimeError(training_status)
                if training_status == "COMPLETED":
                    break
            if args.once or stopping:
                break
            time.sleep(args.poll_seconds)
        state["status"] = "stopped"
        save_state(state_file, state)
    lock.close()


if __name__ == "__main__":
    main()
