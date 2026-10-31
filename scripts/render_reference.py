#!/usr/bin/env python3
"""Render a 50 Hz reference from two views at real-time playback speed."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from supermimic.data.motion_convert import g1_xml_path
from supermimic.data.quality import validate_reference


def render(reference, output, title):
    with np.load(reference, allow_pickle=False) as a:
        validate_reference(a)
        root = list(a["body_names"]).index("pelvis")
        poses = np.c_[a["body_pos_w"][:, root], a["body_quat_w"][:, root], a["joint_pos"]]
        times = a["timestamps"].copy()
    spec = mujoco.MjSpec.from_file(str(g1_xml_path()))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[100, 100, .1], rgba=[.65, .68, .72, 1])
    spec.worldbody.add_light(pos=[3, -3, 5], dir=[-.4, .4, -1],
        diffuse=[.8, .8, .8], ambient=[.35, .35, .35])
    spec.worldbody.add_light(pos=[-3, 2, 4], dir=[.4, -.3, -1],
        diffuse=[.6, .6, .6], castshadow=False)
    model = spec.compile()
    data = mujoco.MjData(model)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    cameras = []
    for azimuth in (135, 45):
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3., azimuth, -12
        cameras.append(camera)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    indices = list(range(0, len(poses), 2))  # 50 Hz / 2 = 25 fps, no time stretch.
    snapshot_index = indices[len(indices)//2]
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(output, fps=25, codec="libx264", quality=8) as writer:
            for i in indices:
                data.qpos[:] = poses[i]
                mujoco.mj_forward(model, data)
                views = []
                for camera in cameras:
                    camera.lookat[:] = [poses[i, 0], poses[i, 1], .85]
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    views.append(renderer.render().copy())
                canvas = Image.fromarray(np.concatenate(views, axis=1))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 1280, 58), fill=(25, 28, 33))
                draw.text((12, 5), f"{title} | G1 native 23 DoF | SOMA", font=font, fill="white")
                draw.text((12, 31), f"t = {times[i]:.2f}s / {times[-1]:.2f}s | 1x speed | Kinematic reference", font=font, fill=(220, 225, 235))
                frame = np.asarray(canvas)
                writer.append_data(frame)
                if i == snapshot_index:
                    imageio.imwrite(output.with_suffix(".png"), frame)
    metadata = {"reference": str(Path(reference).resolve()), "video": str(output.resolve()),
        "reference_duration_s": float(times[-1]), "video_fps": 25,
        "video_frames": len(indices), "playback_speed": 1., "kinematic_only": True}
    output.with_suffix(".video.json").write_text(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--title", default="Motion")
    a = p.parse_args()
    print(json.dumps(render(a.reference, a.output, a.title), indent=2))
