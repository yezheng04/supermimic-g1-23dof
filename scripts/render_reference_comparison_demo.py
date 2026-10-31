"""Overlay recorded policy execution with the same-time retargeted reference.

After termination, the gray robot explicitly freezes at its terminal pose while
the complete reference continues. No dynamics or world-model predictions are used
to invent post-termination execution. Coordinates are never fitted or realigned.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_world_model_demo import overlay_model
from supermimic.data.packed_library import load_pack, sha256
from supermimic.robots.spec import NQ


def main(args):
    source, out = args.source, args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    recording = json.loads((source / "recording.json").read_text())
    trajectories = np.load(source / "trajectories.npz", allow_pickle=False)
    pack = load_pack(source / "motions.pt")
    if pack["names"] != trajectories["names"].tolist():
        raise ValueError("Reference and recorded clip order differ")
    if int(trajectories["fps"]) != 50 or pack["fps"] != 50:
        raise ValueError("Expected a shared 50 Hz timeline")
    model = overlay_model(ghost_count=1, white_stage=True)
    model.vis.global_.offwidth = 1600
    model.vis.global_.offheight = 900
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=900, width=1600)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[:3] = 1
    option.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 140, -24
    fonts = {n: ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", n)
             for n in (18, 22, 26, 34)}
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    outputs, clips = [], []
    total_frames = 0
    try:
        for i, row in enumerate(recording["clips"]):
            start, length = int(pack["starts"][i]), int(pack["lengths"][i])
            reference = pack["ref"][start:start + length, :NQ].float().numpy().copy()
            reference[:, 3:7] /= np.linalg.norm(reference[:, 3:7], axis=1, keepdims=True)
            steps = int(trajectories["steps"][i])
            actual = trajectories["states"][:steps + 1, i, :NQ]
            if length != row["reference_frames"] or len(actual) != steps + 1:
                raise ValueError("Clip duration mismatch")
            initial_position_error = float(np.linalg.norm(actual[0, :3] - reference[0, :3]))
            initial_joint_error = float(np.max(np.abs(actual[0, 7:] - reference[0, 7:])))
            if initial_position_error > .001 or initial_joint_error > .001:
                raise ValueError("Initial actual/reference poses do not match; check coordinate frames")
            indices = list(range(0, length, 2))  # 50 Hz -> 25 fps, original speed.
            path = out / "clips" / f"{i+1:02d}_{row['name']}.mp4"
            proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", "1600x900", "-r", "25", "-i", "-", "-an",
                "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
            try:
                for t in indices:
                    frozen = t > steps
                    current = actual[min(t, steps)]
                    target = reference[t]
                    data.qpos[:NQ], data.qpos[NQ:] = current, target
                    center = (current[:3] + target[:3]) / 2
                    camera.lookat[:] = [center[0], center[1], .42]
                    camera.distance = max(2.8, 2.4 + float(np.linalg.norm(current[:3] - target[:3])))
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera, option)
                    image = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(image)
                    draw.text((40, 24), row["title"] + " · 策略与参考", font=fonts[34], fill=(25, 30, 38))
                    draw.text((40, 76), "G1 23DoF  /  Checkpoint 80,000  /  MuJoCo 仿真", font=fonts[22], fill=(100, 110, 125))
                    draw.text((40, 112), row["name"], font=fonts[18], fill=(110, 120, 130))
                    draw.ellipse((1050, 35, 1066, 51), fill=(105, 110, 116))
                    draw.text((1080, 24), "策略执行", font=fonts[26], fill=(55, 60, 68))
                    draw.ellipse((1280, 35, 1296, 51), fill=(30, 155, 240))
                    draw.text((1310, 24), "重定向参考", font=fonts[26], fill=(55, 60, 68))
                    if frozen:
                        status = f"策略于 {steps/50:.2f} s 终止：灰色固定为终止姿态，仅蓝色参考继续播放"
                        draw.text((40, 748), status, font=fonts[26], fill=(185, 65, 50))
                        detail = "终止原因：" + row["failure_reason"] + "  ·  此后不计算同步跟踪误差"
                    else:
                        root_error = np.linalg.norm(current[:3] - target[:3]) * 100
                        joint_error = np.sqrt(np.mean((current[7:] - target[7:]) ** 2)) * 180 / np.pi
                        draw.text((40, 748), "同一时间戳叠加 · 保留原始位置与朝向", font=fonts[26], fill=(65, 80, 100))
                        detail = f"根部位置误差 {root_error:.1f} cm  /  关节 RMSE {joint_error:.1f}°"
                    draw.text((40, 791), detail, font=fonts[22], fill=(80, 90, 104))
                    draw.line((40, 840, 1560, 840), fill=(222, 227, 233), width=3)
                    draw.line((40, 840, 40 + 1520*t/(length-1), 840), fill=(42, 146, 225), width=3)
                    failure_x = 40 + 1520*steps/(length-1)
                    draw.line((failure_x, 832, failure_x, 848), fill=(185, 65, 50), width=3)
                    draw.text((40, 855), f"{t/50:.2f} s / {(length-1)/50:.2f} s  ·  50Hz 时间轴 / 25fps 视频  ·  1×  ·  红线：策略终止", font=fonts[18], fill=(100, 108, 119))
                    if t in (2*(steps//4), indices[len(indices)//2]):
                        image.save(out / "previews" / f"{i+1:02d}_{'after' if frozen else 'before'}_termination.png")
                    proc.stdin.write(np.asarray(image).tobytes())
            finally:
                proc.stdin.close()
                code = proc.wait()
            if code:
                raise RuntimeError("Video encoding failed")
            outputs.append(path)
            total_frames += len(indices)
            clips.append(dict(name=row["name"], failure_time_s=steps/50, failure_reason=row["failure_reason"],
                reference_duration_s=(length-1)/50, video_frames=len(indices),
                initial_root_error_m=initial_position_error, initial_joint_max_error_rad=initial_joint_error))
            print(f"Rendered {i+1}/{len(recording['clips'])}: {row['name']}", flush=True)
        listing = out / "concat.txt"
        listing.write_text("".join(f"file '{p.resolve()}'\n" for p in outputs))
        final = out / args.video_name
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
        subprocess.run([ffmpeg, "-v", "error", "-i", str(final), "-f", "null", "-"], check=True)
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(final))
        if frames != total_frames:
            raise ValueError("Encoded video frame count differs")
        report = dict(path=str(final), sha256=sha256(final), fps=25, width=1600, height=900,
            frames=frames, seconds=seconds, checkpoint=recording["checkpoint"],
            trajectory_sha256=sha256(source / "trajectories.npz"), reference_sha256=sha256(source / "motions.pt"),
            mode="Same-time actual simulator policy execution versus retargeted reference; no world-model predictions",
            coordinates="Shared environment-local coordinates, no spatial or temporal realignment",
            after_termination="Gray robot freezes at the exact captured pre-reset terminal pose, explicitly labeled; blue reference continues",
            timebase="50 Hz source, every second frame rendered at 25 fps; no speedup; endpoint quantization <=20 ms",
            clips=clips)
        (out / "video.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(final, flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/demo_80k_side_rolls_20261030"))
    parser.add_argument("--output", type=Path, default=Path("reports/demo_80k_side_rolls_reference_20261030"))
    parser.add_argument("--video-name", default="g1_23dof_80k_side_roll_policy_vs_reference.mp4")
    main(parser.parse_args())
