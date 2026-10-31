#!/usr/bin/env python3
"""Newton OpenGL playback of native23 references and verified SOMA overlays."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", choices=["wave", "hiphop", "jump", "pickup", "throw", "stretch"], default="wave")
    parser.add_argument("--reference", type=Path, help="Alternative reference_50hz.npz")
    parser.add_argument("--overlay", type=Path, help="Directory containing verified overlay.npz and overlay.json")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N rendered frames (0: interactive)")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.headless:
        os.environ["PYGLET_HEADLESS"] = "1"
        if args.frames <= 0:
            parser.error("--headless requires --frames > 0")
    elif not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("No desktop display. Run this launcher in a terminal inside your graphical desktop session.")

    import numpy as np
    import warp as wp
    import newton
    from newton.viewer import ViewerGL
    from scipy.spatial.transform import Rotation, Slerp
    from supermimic.robots.spec import JOINT_NAMES
    from supermimic.data.quality import validate_reference

    model_path = ROOT / "src/supermimic/robots/g1_23dof.xml"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    examples = ROOT / "data/retargeted/soma_examples_v1"
    overlays = ROOT / "data/retargeted/soma_examples_overlay_v1"
    entries = json.loads((examples / "summary.json").read_text())
    if args.reference:
        entries = [dict(motion=args.reference.parent.name, title=args.reference.parent.name)]
    elif args.overlay:
        parser.error("--overlay requires --reference")

    wp.set_device("cuda:0")
    builder = newton.ModelBuilder()
    builder.add_mjcf(str(model_path))
    builder.add_ground_plane()
    model = builder.finalize()
    if model.joint_coord_count != 30 or tuple(x.split("/")[-1] for x in model.joint_label[1:]) != JOINT_NAMES:
        raise ValueError("Newton model is not the expected native23 joint order")
    state = model.state()
    body_names = [x.split("/")[-1] for x in model.body_label]
    viewer = ViewerGL(width=1440, height=900, vsync=True, headless=args.headless)
    viewer.renderer.set_title("SOMA / G1 native23 - reference and human overlay")
    viewer.set_model(model)
    viewer.set_world_offsets([0.0, 0.0, 0.0])

    class Playback:
        def __init__(self):
            self.index = 0 if args.reference else next(i for i, e in enumerate(entries) if e["motion"] == args.motion)
            self.playing, self.loop, self.speed = True, True, 1.0
            self.source, self.targets, self.errors = True, True, True
            self.load()

        def load(self):
            entry = entries[self.index]
            reference = args.reference or examples / entry["motion"] / "reference_50hz.npz"
            meta = json.loads(reference.with_suffix(".json").read_text())
            if meta["model_sha256"] != model_hash:
                raise ValueError("Reference model hash differs from native23 XML")
            with np.load(reference, allow_pickle=False) as ref:
                validate_reference(ref)
                if tuple(ref["joint_names"].tolist()) != JOINT_NAMES:
                    raise ValueError("Reference joint order mismatch")
                root = ref["body_names"].tolist().index("pelvis")
                self.times = ref["timestamps"].copy()
                self.q = np.c_[ref["body_pos_w"][:, root], ref["body_quat_w"][:, root][:, [1, 2, 3, 0]], ref["joint_pos"]]
            self.rotations = Slerp(self.times, Rotation.from_quat(self.q[:, 3:7]))
            self.overlay = None
            directory = args.overlay if args.reference else overlays / entry["motion"]
            if directory:
                info = json.loads((directory / "overlay.json").read_text())
                if info["reference_sha256"] != hashlib.sha256(reference.read_bytes()).hexdigest():
                    raise ValueError("Overlay belongs to a different reference")
                with np.load(directory / "overlay.npz", allow_pickle=False) as ov:
                    self.overlay = {k: ov[k].copy() for k in ov.files}
                if not np.array_equal(self.times, self.overlay["timestamps"]):
                    raise ValueError("Overlay timestamps mismatch")
                self.target_indices = [body_names.index(x) for x in self.overlay["target_bodies"]]
            self.t = float(self.times[0])
            self.reset_camera()
            print(f"Loaded {entry['title']}: {len(self.times)} frames, {self.times[-1]:.2f}s, native23", flush=True)

        def reset_camera(self):
            p = self.q[0, :3]
            viewer.set_camera(wp.vec3(float(p[0] + 2), float(p[1] - 2), 1.5), pitch=-12, yaw=135)

        def gui(self, ui):
            ui.set_next_window_size(ui.ImVec2(410, 330), ui.Cond_.first_use_ever)
            ui.begin("Native23 motion playback")
            changed, index = ui.combo("Motion", self.index, [e["title"] for e in entries])
            if changed:
                self.index = index
                self.load()
            if ui.button("Pause" if self.playing else "Play"):
                self.playing = not self.playing
            ui.same_line()
            if ui.button("Restart"):
                self.t = float(self.times[0])
            ui.same_line()
            if ui.button("Reset camera"):
                self.reset_camera()
            changed, self.t = ui.slider_float("Time (s)", self.t, float(self.times[0]), float(self.times[-1]), "%.3f")
            if changed:
                self.playing = False
            _, self.speed = ui.slider_float("Speed", self.speed, 0.1, 2.0, "%.2fx")
            _, self.loop = ui.checkbox("Loop", self.loop)
            if self.overlay is not None:
                _, self.source = ui.checkbox("Raw human skeleton (cyan, unscaled)", self.source)
                _, self.targets = ui.checkbox("Calibrated IK targets (orange)", self.targets)
                _, self.errors = ui.checkbox("Target to robot errors (red)", self.errors)
            else:
                ui.text("Robot only: pass --overlay for source comparison")
            ui.text("50 Hz reference; playback uses elapsed seconds")
            ui.text("Kinematic playback, not a tracking controller")
            ui.end()

        def render(self):
            i = int(np.clip(np.searchsorted(self.times, self.t, side="right") - 1, 0, len(self.times) - 2))
            alpha = (self.t - self.times[i]) / (self.times[i + 1] - self.times[i])
            def sample(x):
                return (1 - alpha) * x[i] + alpha * x[i + 1]
            q = sample(self.q)
            q[3:7] = self.rotations(self.t).as_quat()
            model.joint_q.assign(q.astype(np.float32))
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            viewer.begin_frame(self.t)
            viewer.log_state(state)
            def vec(x):
                return wp.array(np.asarray(x, dtype=np.float32), dtype=wp.vec3, device=model.device)
            if self.overlay is not None:
                for key, enabled, color in (("source", self.source, (0.1, 0.85, 1.0)), ("target", self.targets, (1.0, 0.55, 0.05))):
                    points = sample(self.overlay[key + "_pos"])
                    edges = self.overlay[key + "_edges"]
                    viewer.log_lines(key, vec(points[edges[:, 0]]) if enabled else None,
                                     vec(points[edges[:, 1]]) if enabled else None, color, width=0.008)
                    viewer.log_points(key + "_joints", vec(points) if enabled else None,
                                      radii=wp.full(len(points), 0.015, dtype=wp.float32, device=model.device),
                                      colors=vec(np.tile(color, (len(points), 1))))
                target = sample(self.overlay["target_pos"])
                actual = state.body_q.numpy()[self.target_indices, :3]
                viewer.log_lines("errors", vec(target) if self.errors else None,
                                 vec(actual) if self.errors else None, (1.0, 0.1, 0.12), width=0.005)
            viewer.end_frame()

    try:
        playback = Playback()
        viewer.register_ui_callback(playback.gui, position="free")
        previous = time.monotonic()
        frame = 0
        while viewer.is_running() and (not args.frames or frame < args.frames):
            now = time.monotonic()
            dt, previous = now - previous, now
            if playback.playing:
                playback.t += (1 / 60 if args.headless else dt) * playback.speed
                if playback.t > playback.times[-1]:
                    if playback.loop:
                        playback.t = float(playback.times[0] + (playback.t - playback.times[0]) % (playback.times[-1] - playback.times[0]))
                    else:
                        playback.t, playback.playing = float(playback.times[-1]), False
            playback.render()
            frame += 1
        if args.screenshot:
            from PIL import Image
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
            print(f"Saved {args.screenshot}", flush=True)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
