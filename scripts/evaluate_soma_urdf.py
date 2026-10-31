#!/usr/bin/env python3
"""Common-model/timebase diagnostics and synchronized native23 comparison videos."""
import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic.data.motion_convert import load_g1_model, g1_xml_path
from supermimic.data.quality import validate_reference
from supermimic.retarget.bvh import read_bvh_timing


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def contacts(source, times):
    import warp as wp
    from soma_retargeter.io.bvh import load_bvh
    from soma_retargeter.animation.animation_buffer import AnimationBuffer
    from soma_retargeter.animation.contact_detection import BaselineAThresholds, detect_contacts_velocity_jerk
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    config = json.loads((ROOT.parent / "SOMA-Retargeter-official/soma_retargeter/assets/soma/contact_processing/soma_contact_config.json").read_text())
    wp.set_device("cpu")
    skeleton, animation = load_bvh(str(source))
    n, fps = read_bvh_timing(source)
    animation = AnimationBuffer(skeleton, n, fps, animation.local_transforms)
    thresholds = BaselineAThresholds(**{f.name: config[f.name] for f in fields(BaselineAThresholds) if f.name in config})
    result = detect_contacts_velocity_jerk(skeleton, animation,
        SpaceConverter(FacingDirectionType.MUJOCO).transform(wp.transform_identity()), config["joint_names"], thresholds)
    ids = np.clip(np.rint(times * fps).astype(int), 0, n-1)
    return result.contacts[ids][:, [0, 2]] >= .999


def tracking(ref, overlay):
    bn = ref["body_names"].tolist()
    names = overlay["target_names"].tolist()
    actual = ref["body_pos_w"][:, [bn.index(x) for x in overlay["target_bodies"]]]
    actual_r = ref["body_quat_w"][:, [bn.index(x) for x in overlay["orientation_bodies"]]][:, :, [1, 2, 3, 0]]
    delta = actual - overlay["target_pos"]
    pos = np.linalg.norm(delta, axis=-1) * 100
    rel = np.linalg.norm(delta - delta[:, names.index("Hips"), None], axis=-1) * 100
    angle = np.rad2deg((Rotation.from_quat(actual_r.reshape(-1, 4)).inv() *
                       Rotation.from_quat(overlay["target_quat_xyzw"].reshape(-1, 4))).magnitude().reshape(pos.shape))
    groups = dict(all=list(range(len(names))), hands=[names.index(x) for x in ("LeftHand", "RightHand")],
                  feet=[names.index(x) for x in ("LeftFoot", "RightFoot")], pelvis=[names.index("Hips")],
                  chest=[names.index("Chest")])
    return {name: dict(mean_position_cm=float(pos[:, ids].mean()), p95_position_cm=float(np.quantile(pos[:, ids], .95)),
                      max_position_cm=float(pos[:, ids].max()), mean_root_relative_position_cm=float(rel[:, ids].mean()),
                      mean_rotation_deg=float(angle[:, ids].mean()),
                      first_frame_position_cm=float(pos[0, ids].mean()), last_frame_position_cm=float(pos[-1, ids].mean()))
            for name, ids in groups.items()}


def diagnose(ref, report, mask, model):
    bn = ref["body_names"].tolist()
    foot_ids = [bn.index(x + "_ankle_roll_link") for x in ("left", "right")]
    foot_p = ref["body_pos_w"][:, foot_ids]
    foot_r = Rotation.from_quat(ref["body_quat_w"][:, foot_ids][:, :, [1, 2, 3, 0]].reshape(-1, 4))
    # A fixed sole-centre point in the same canonical model for every candidate.
    sole = foot_p + foot_r.apply(np.tile([.04, 0, -.035], (len(foot_p)*2, 1))).reshape(foot_p.shape)
    speed = np.linalg.norm(np.gradient(sole, .02, axis=0)[:, :, :2], axis=-1)[mask]
    q = ref["joint_pos"]
    margin = np.minimum(q - model.jnt_range[1:, 0], model.jnt_range[1:, 1] - q)
    acceleration = np.gradient(ref["joint_vel"], .02, axis=0)
    quality = report["quality"]
    return dict(frames=len(q), duration_s=float(ref["timestamps"][-1]),
        collision_frames=quality["self_collision_frames_2mm"],
        collision_fraction=quality["self_collision_frames_2mm"]/len(q),
        penetration_frames=quality["foot_penetration_frames_2cm"],
        penetration_fraction=quality["foot_penetration_frames_2cm"]/len(q),
        minimum_foot_z_m=quality["minimum_foot_surface_z_m"],
        limit_violation_frames=quality["joint_limit_violation_frames"],
        near_limit_fraction=float(np.mean(np.any(margin < np.deg2rad(1), axis=-1))),
        p95_joint_acceleration_rad_s2=float(np.quantile(np.abs(acceleration), .95)),
        max_joint_acceleration_rad_s2=float(np.max(np.abs(acceleration))),
        max_joint_speed_rad_s=float(np.max(np.abs(ref["joint_vel"]))),
        source_contact_samples=int(mask.sum()),
        source_contact_sole_xy_speed_mean_m_s=float(speed.mean()) if len(speed) else None,
        source_contact_sole_xy_speed_p95_m_s=float(np.quantile(speed, .95)) if len(speed) else None,
        source_contact_sole_speed_over_10cm_fraction=float(np.mean(speed > .1)) if len(speed) else None)


def render(row, refs, overlays, output):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    spec = mujoco.MjSpec.from_file(str(g1_xml_path()))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[100,100,.1], rgba=[.19,.23,.29,1])
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    model.geom_rgba[model.geom_bodyid != 0, 3] = .8
    model.mat_rgba[:, 3] = .8
    model.vis.headlight.ambient[:] = [.5, .5, .5]
    model.vis.headlight.diffuse[:] = [.8, .8, .8]
    data = mujoco.MjData(model)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.4, 135, -12
    poses = []
    for ref in refs:
        root = ref["body_names"].tolist().index("pelvis")
        poses.append(np.c_[ref["body_pos_w"][:, root], ref["body_quat_w"][:, root], ref["joint_pos"]])
    labels = ["A: Current community + local offset", "B: Own URDF / official defaults", "C: Own URDF / 2 optimizer iterations"]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    times = refs[0]["timestamps"]
    indices = list(range(0, len(times), 2))
    output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer, imageio.get_writer(output, fps=25, codec="libx264", quality=8) as writer:
        for i in indices:
            cam.lookat[:] = [np.mean([q[i, 0] for q in poses]), np.mean([q[i, 1] for q in poses]), .85]
            views = []
            for q, ov in zip(poses, overlays):
                data.qpos[:] = q[i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam, scene_option=opt)
                for j, body in enumerate(ov["target_bodies"]):
                    target = ov["target_pos"][i, j]
                    actual = data.xpos[model.body(str(body)).id]
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, [.014]*3, target, np.eye(3).ravel(), [1,.6,.06,1])
                    renderer.scene.ngeom += 1
                    if np.linalg.norm(target-actual) > 1e-8:
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), [1,.1,.2,1])
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .004, actual, target)
                        renderer.scene.ngeom += 1
                views.append(renderer.render().copy())
            canvas = Image.new("RGB", (1920, 560), (25,28,33))
            canvas.paste(Image.fromarray(np.concatenate(views, axis=1)), (0, 48))
            draw = ImageDraw.Draw(canvas)
            for col, label in enumerate(labels):
                draw.text((col*640+10, 4), label, font=font, fill="white")
                draw.text((col*640+10, 25), f"{row['motion']} | {times[i]:.2f}/{times[-1]:.2f}s | 1x", font=font, fill="white")
            draw.text((12,535), "Same time / camera / native23 model. Orange: each config's own pre-IK targets. Red: position errors. Kinematic playback.", font=font, fill="white")
            frame = np.asarray(canvas)
            writer.append_data(frame)
            if i == indices[len(indices)//2]:
                imageio.imwrite(output.with_suffix(".png"), frame)
    return dict(video=str(output), frames=len(indices), fps=25, duration_s=len(indices)/25)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "data/retargeted/soma_urdf_comparison_v1")
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    rows = json.loads((args.root / "manifest.json").read_text())
    assert len(rows) == 10
    model = load_g1_model()
    results, videos = [], []
    for row in rows:
        variants = dict(current=Path(row["current"]), official=Path(row["official"]), optimized=args.root / "optimized" / row["motion"])
        if (args.root / "no_contact" / row["motion"]).is_dir():
            variants["no_contact"] = args.root / "no_contact" / row["motion"]
        refs = {key: load(path / "reference_50hz.npz") for key, path in variants.items()}
        reports = {key: json.loads((path / "reference_50hz.json").read_text()) for key, path in variants.items()}
        overlays = {key: load((Path(row["current_overlay"]) if key == "current" else path / "overlay") / "overlay.npz") for key, path in variants.items()}
        for key, ref in refs.items():
            validate_reference(ref)
            assert np.array_equal(ref["timestamps"], refs["current"]["timestamps"])
            assert np.array_equal(ref["timestamps"], overlays[key]["timestamps"])
            assert reports[key]["source_sha256"] == row["source_sha256"]
        assert reports["official"]["human_scaler_config"] == reports["optimized"]["human_scaler_config"]
        mask = contacts(Path(row["source"]), refs["current"]["timestamps"])
        np.savez_compressed(args.root / "official" / row["motion"] / "source_contact_mask.npz", timestamps=refs["current"]["timestamps"], mask=mask)
        result = dict(motion=row["motion"], split=row["split"], variants={})
        for key, ref in refs.items():
            result["variants"][key] = dict(quality=diagnose(ref, reports[key], mask, model),
                tracking_against_current_targets=tracking(ref, overlays["current"]),
                tracking_against_urdf_targets=tracking(ref, overlays["official"]))
        results.append(result)
        if args.render:
            keys = ("current", "official", "optimized")
            videos.append(render(row, [refs[k] for k in keys], [overlays[k] for k in keys], args.root / "videos" / (row["motion"] + ".mp4")))
        print("Evaluated", row["motion"], flush=True)
    summary = {}
    for split in ("all", "calibration", "holdout"):
        selected = [r for r in results if split == "all" or r["split"] == split]
        summary[split] = {}
        for variant in ("current", "official", "optimized"):
            vals = [r["variants"][variant] for r in selected]
            total = sum(v["quality"]["frames"] for v in vals)
            summary[split][variant] = dict(motions=len(vals), frames=total,
                collision_fraction=sum(v["quality"]["collision_frames"] for v in vals)/total,
                penetration_fraction=sum(v["quality"]["penetration_frames"] for v in vals)/total,
                mean_clip_p95_joint_acceleration_rad_s2=float(np.mean([v["quality"]["p95_joint_acceleration_rad_s2"] for v in vals])),
                mean_clip_contact_sole_speed_m_s=float(np.mean([v["quality"]["source_contact_sole_xy_speed_mean_m_s"] for v in vals if v["quality"]["source_contact_sole_xy_speed_mean_m_s"] is not None])),
                mean_clip_hand_error_to_current_targets_cm=float(np.mean([v["tracking_against_current_targets"]["hands"]["mean_position_cm"] for v in vals])),
                mean_clip_hand_error_to_urdf_targets_cm=float(np.mean([v["tracking_against_urdf_targets"]["hands"]["mean_position_cm"] for v in vals])))
    data = dict(summary=summary, motions=results, videos=videos,
        metrics_note="All outputs evaluated with identical canonical MJCF at exact 50Hz timestamps. Collision is proxy-model penetration, not mesh or dynamic certification. Foot speed uses a fixed sole-centre point during official source-detected contact (nearest source sample, un-ramped contact weight>=0.999), not ground-truth force labels. Tracking is cross-evaluated against BOTH target sets; do not compare diagonal losses as a controlled calibration improvement. A/B changes pipeline, calibration and runtime; B/C only changes IK weights.",
        scope="10 deliberately selected official examples, not a representative full-dataset estimate; optimizer only sees four calibration motions")
    (args.root / "comparison.json").write_text(json.dumps(data, indent=2))
    if args.render:
        import imageio.v2 as imageio
        with imageio.get_writer(args.root / "all_motions_comparison.mp4", fps=25, codec="libx264", quality=8) as writer:
            for video in videos:
                with imageio.get_reader(video["video"]) as reader:
                    for frame in reader:
                        writer.append_data(frame)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
