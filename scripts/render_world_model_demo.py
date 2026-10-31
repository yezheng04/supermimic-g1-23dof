"""Render paired real-simulator/world-model overlays from recorded trajectories."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supermimic.data.motion_convert import g1_xml_path
from supermimic.robots.spec import NQ


def overlay_model(ghost_count=1, white_stage=False):
    path = g1_xml_path()
    root = ET.fromstring(path.read_text())
    root.find("compiler").set("meshdir", str((path.parent / "assets").resolve()))
    for tag in ("actuator", "sensor", "keyframe", "contact", "visual"):
        for el in list(root.findall(tag)):
            root.remove(el)
    world = root.find("worldbody")
    original = world.find("body")
    ghost = copy.deepcopy(original)
    for parent in ghost.iter():
        for child in list(parent):
            if child.tag in ("camera", "light", "site") or (child.tag == "geom" and "mesh" not in child.attrib):
                parent.remove(child)
        if "name" in parent.attrib:
            parent.set("name", "wm_" + parent.get("name"))
        if parent.tag == "geom":
            for key in ("class", "material", "rgba"):
                parent.attrib.pop(key, None)
            parent.attrib.update(type="mesh", contype="0", conaffinity="0", density="0",
                                 group="1", rgba="0.10 0.58 1.0 0.45")
    for body in original.iter("body"):
        for light in list(body.findall("light")):
            body.remove(light)
    for index in range(ghost_count):
        duplicate = copy.deepcopy(ghost)
        if ghost_count > 1:
            for element in duplicate.iter():
                if "name" in element.attrib:
                    element.set("name", f"future_{index}_" + element.get("name"))
                if element.tag == "geom":
                    alpha = .28 * (1.0-index/ghost_count)**1.5 + .015
                    element.set("rgba", f"0.06 0.56 0.96 {alpha:.4f}")
        world.append(duplicate)
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="demo_sky", type="skybox", builtin="gradient", width="512", height="512",
                  rgb1="0.68 0.76 0.86", rgb2="0.92 0.94 0.97")
    ET.SubElement(asset, "texture", name="demo_grid", type="2d", builtin="checker", width="512", height="512",
                  rgb1="0.92 0.94 0.96", rgb2="0.82 0.85 0.89", mark="edge", markrgb="0.72 0.76 0.82")
    ET.SubElement(asset, "material", name="demo_floor", texture="demo_grid", texrepeat="2 2", texuniform="true", reflectance="0", specular="0.05")
    ET.SubElement(world, "geom", name="demo_floor", type="plane", size="0 0 .05", material="demo_floor", contype="0", conaffinity="0", group="0")
    ET.SubElement(world, "light", name="key", pos="3 -3 5", dir="-.4 .4 -1", diffuse=".5 .5 .5", ambient=".2 .2 .2")
    ET.SubElement(world, "light", name="fill", pos="-3 2 4", dir=".4 -.3 -1", diffuse=".3 .3 .3", castshadow="false")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="800", offheight="640")
    ET.SubElement(vis, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(vis, "map", znear="0.01")
    ET.SubElement(vis, "headlight", ambient=".2 .2 .2", diffuse=".3 .3 .3", specular=".05 .05 .05")
    if white_stage:
        sky = asset.find("texture[@name='demo_sky']")
        sky.set("rgb1", "1 1 1"); sky.set("rgb2", "1 1 1")
        floor = asset.find("material[@name='demo_floor']")
        floor.attrib.pop("texture", None)
        floor.set("rgba", "1 1 1 1")
        key = world.find("light[@name='key']")
        key.set("ambient", ".5 .5 .5"); key.set("directional", "true")
        fill = world.find("light[@name='fill']")
        fill.set("diffuse", ".5 .5 .5"); fill.set("directional", "true")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.nq == (ghost_count+1)*NQ
    return model


def metrics(actual, predicted):
    if not np.isfinite(predicted).all():
        return "预测数值已发散", False
    root_cm = np.linalg.norm(actual[:3] - predicted[:3])*100
    joint_deg = np.sqrt(np.mean((actual[7:NQ] - predicted[7:NQ])**2))*180/np.pi
    return f"根部位置误差 {root_cm:.1f} cm   |   关节 RMSE {joint_deg:.1f}°", root_cm < 1000


def main(args):
    folder = args.folder
    d = np.load(folder / "trajectories.npz", allow_pickle=False)
    report = json.loads((folder / "recording.json").read_text())
    model = overlay_model()
    data = mujoco.MjData(model)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[:3] = 1
    opt.sitegroup[:] = 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 140, -16, 2.8
    renderer = mujoco.Renderer(model, height=640, width=800)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fonts = {size: ImageFont.truetype(font_path, size) for size in (18, 21, 24, 28, 36)}
    video_dir = folder / "clips"
    video_dir.mkdir(exist_ok=True)
    preview_dir = folder / "previews"
    preview_dir.mkdir(exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    paths = []
    width, height, fps = 1600, 900, 25
    try:
        for i, name in enumerate(d["names"].tolist()):
            steps = int(d["steps"][i])
            row = report["clips"][i]
            path = video_dir / f"{i+1:02d}_{name}.mp4"
            if args.preview_only:
                indices = [min(steps, max(2, steps//2))]
                proc = None
            else:
                indices = list(range(0, steps+1, 2))  # exact 50Hz -> 25fps decimation, no motion speedup
                proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-threads", "4",
                    "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
            target = d["states"][0, i, :3].copy()
            targets = []
            for frame in range(steps+1):
                target = .936*target + .064*d["states"][frame, i, :3]
                targets.append(target.copy())
            for t in indices:
                actual = d["states"][t, i]
                target = targets[t]
                panels, labels, visible = [], [], []
                for key in ("wm_32", "wm_continuous"):
                    pred = d[key][t, i]
                    label, show = metrics(actual, pred)
                    cam.lookat[:] = [target[0], target[1], .48 if actual[2] < .5 else .65]
                    cam.distance = 2.8
                    if key == "wm_continuous" and show:
                        center = (actual[:3] + pred[:3]) / 2
                        cam.lookat[:] = [center[0], center[1], max(.4, min(1.2, center[2]))]
                        cam.distance = max(2.8, 2.2 + 1.35*np.linalg.norm(actual[:3]-pred[:3]))
                    data.qpos[:NQ] = actual[:NQ]
                    if show:
                        data.qpos[NQ:] = pred[:NQ]
                    else:
                        data.qpos[NQ:] = actual[:NQ]
                        data.qpos[NQ:NQ+3] = [0, 0, -50]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, cam, opt)
                    panels.append(renderer.render().copy())
                    labels.append(label); visible.append(show)
                image = Image.new("RGB", (width, height), (17, 24, 39))
                image.paste(Image.fromarray(panels[0]), (0, 130))
                image.paste(Image.fromarray(panels[1]), (800, 130))
                draw = ImageDraw.Draw(image)
                draw.text((26, 12), f"G1 · 23DoF    {row['title']}", font=fonts[36], fill="white")
                draw.text((26, 62), "Checkpoint 80,000  ·  MuJoCo Warp 仿真  ·  控制 50Hz  ·  播放 1×", font=fonts[21], fill=(190,205,225))
                draw.text((26, 96), name, font=fonts[18], fill=(155,175,200))
                draw.ellipse((1090,24,1108,42), fill=(180,180,180))
                draw.text((1118,18), "实体：策略仿真执行", font=fonts[21], fill="white")
                draw.ellipse((1090,60,1108,78), fill=(45,165,255))
                draw.text((1118,54), "蓝色：世界模型预测", font=fonts[21], fill=(105,195,255))
                age = 0 if t == 0 else (t-1)%32+1
                for panel, title, subtitle in ((0,"32步开环预测",f"每0.64秒重置起点  ·  当前预测 {age}/32 步"),
                                               (1,"整段连续预测","仅首帧初始化  ·  之后不读取仿真状态")):
                    x = panel*800
                    draw.rounded_rectangle((x+16,146,x+780,218), radius=10, fill=(22,34,52))
                    draw.text((x+30,149),title,font=fonts[28],fill="white")
                    draw.text((x+30,185),subtitle,font=fonts[21],fill=(175,205,235))
                    draw.text((x+24,787),labels[panel],font=fonts[21],fill=(140,210,255))
                    if not visible[panel]:
                        draw.text((x+24,720),"预测发散 / 超出展示范围",font=fonts[24],fill=(255,95,95))
                draw.line((800,130,800,770),fill=(17,24,39),width=4)
                status = "失败终止" if row["failed"] else "完整播放"
                draw.text((26,824),f"t = {t/50:.2f} s / {steps/50:.2f} s    本次执行：{status}    完整回报 {row['full_clip_norm_return']:.3f}",font=fonts[21],fill=(235,190,130) if row["failed"] else (200,215,230))
                draw.text((26,860),"同一组仿真实际动作输入；蓝色不是参考动作。原始坐标叠加，右侧相机自动拉远展示漂移。",font=fonts[21],fill=(160,180,205))
                if args.preview_only or t == indices[len(indices)//2]:
                    image.save(preview_dir / f"{i+1:02d}.png")
                if proc:
                    proc.stdin.write(np.asarray(image).tobytes())
            if proc:
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError(f"Video encoder failed: {path}")
                paths.append(path)
                print(f"Rendered {path.name}: {len(indices)/fps:.2f}s", flush=True)
        if paths:
            listing = folder / "concat.txt"
            listing.write_text("".join(f"file '{p.resolve()}'\n" for p in paths))
            combined = folder / "g1_23dof_80k_sim_vs_world_model.mp4"
            subprocess.run([ffmpeg,"-y","-loglevel","error","-f","concat","-safe","0","-i",str(listing),"-c","copy","-movflags","+faststart",str(combined)],check=True)
            (folder / "video.json").write_text(json.dumps(dict(path=str(combined),width=width,height=height,fps=fps,
                playback_speed=1.0,seconds=sum((int(n)//2+1)/fps for n in d["steps"]),clips=len(paths),
                prediction="32-step reseeded versus continuous open loop; both use executed simulator actions"),indent=2)+"\n")
            print(combined, flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=Path("reports/demo_80k_world_model_20261030"))
    p.add_argument("--preview-only",action="store_true")
    main(p.parse_args())
