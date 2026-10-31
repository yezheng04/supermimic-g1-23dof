#!/usr/bin/env python3
"""Overlay the verified source BVH and calibrated SOMA targets on saved native23 references.

Run in the project venv. Source decoding runs in SOMA's isolated venv on CPU;
no IK is rerun and no per-frame alignment is fitted to the robot trajectory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def prepare(reference, destination):
    import warp as wp
    from soma_retargeter.assets.bvh import load_bvh
    from soma_retargeter.animation.animation_buffer import AnimationBuffer
    from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str
    from supermimic.retarget.bvh import read_bvh_timing
    from supermimic.data.motion_convert import g1_xml_path
    from supermimic.data.quality import validate_reference

    report = json.loads(reference.with_suffix(".json").read_text())
    source = Path(report["source"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != report["source_sha256"]:
        raise ValueError(f"Source BVH changed: {source}")
    if hashlib.sha256(g1_xml_path().read_bytes()).hexdigest() != report["model_sha256"]:
        raise ValueError("Robot model differs from the reference model")
    with np.load(reference, allow_pickle=False) as ref:
        validate_reference(ref)
        times = ref["timestamps"].copy()
    wp.set_device("cpu")
    skeleton, animation = load_bvh(str(source))
    declared, fps = read_bvh_timing(source)
    n = report["source_frames_used"]
    if declared != animation.num_frames or n > declared or not np.isclose(fps, report["source_fps"]):
        raise ValueError("Source frame count or timebase mismatch")
    source_times = np.arange(n, dtype=np.float64) / fps
    if times[-1] > source_times[-1] + 1e-10 or report.get("source_start_s", 0) != 0:
        raise ValueError("Reference lies outside the supported source interval")
    animation = AnimationBuffer(skeleton, n, fps, animation.local_transforms[:n])
    destination.mkdir(parents=True, exist_ok=True)
    scaler_path = destination / "scaler_used.json"
    scaler_path.write_text(json.dumps(report["human_scaler_config"], indent=2))
    scaler = HumanToRobotScaler(skeleton, report["ik_config"]["model_height"], str(scaler_path))
    transform = SpaceConverter(get_facing_direction_type_from_str("Mujoco")).transform(wp.transform_identity())
    targets = scaler.compute_effectors_from_buffer(animation, True, transform)
    names = list(report["ik_config"]["ik_map"])
    target_indices = [scaler.effector_names().index(name) for name in names]
    targets = targets[:, target_indices]

    # Decode source FK in metres, then apply exactly the source-to-Newton transform.
    # Reference offsets are already included in the BVH loader's local transforms.
    local = animation.local_transforms
    global_pos = np.empty((n, skeleton.num_joints, 3), dtype=np.float64)
    global_rot = []
    world_rot = Rotation.from_quat(np.asarray(transform)[3:7])
    world_pos = np.asarray(transform)[:3]
    for j, parent in enumerate(skeleton.parent_indices):
        rotation = Rotation.from_quat(local[:, j, 3:7])
        if parent < 0:
            global_pos[:, j] = world_rot.apply(local[:, j, :3]) + world_pos
            global_rot.append(world_rot * rotation)
        else:
            if parent >= j:
                raise ValueError("BVH joints must be in parent-first order")
            global_pos[:, j] = global_pos[:, parent] + global_rot[parent].apply(local[:, j, :3])
            global_rot.append(global_rot[parent] * rotation)
    # Verify our source FK against SOMA's own FK at start/middle/end.
    fk_error = 0.0
    for frame in (0, n // 2, n - 1):
        upstream = np.asarray(animation.compute_global_transforms(frame, transform))[:, :3]
        fk_error = max(fk_error, float(np.max(np.abs(upstream - global_pos[frame]))))
    if fk_error > 2e-5:
        raise ValueError(f"Source FK disagreement: {fk_error} m")

    source_pos = interp1d(source_times, global_pos, axis=0, bounds_error=True)(times)
    target_pos = interp1d(source_times, targets[:, :, :3], axis=0, bounds_error=True)(times)
    target_quat = np.stack([Slerp(source_times, Rotation.from_quat(targets[:, j, 3:7]))(times).as_quat()
                            for j in range(len(names))], axis=1)
    # Sparse display graph follows each source joint's nearest displayed ancestor.
    source_names = [x for x in skeleton.joint_names if x in set(names) | {
        "Spine", "Spine1", "Spine2", "Neck1", "Neck2", "Head", "HeadEnd",
        "LeftShoulder", "RightShoulder", "LeftToe", "RightToe", "LeftToeBase", "RightToeBase"}]

    def edges(display_names):
        result = []
        for i, name in enumerate(display_names):
            parent = skeleton.parent_indices[skeleton.joint_index(name)]
            while parent >= 0 and skeleton.joint_names[parent] not in display_names:
                parent = skeleton.parent_indices[parent]
            if parent >= 0:
                result.append((display_names.index(skeleton.joint_names[parent]), i))
        return np.array(result, dtype=np.int32).reshape(-1, 2)

    np.savez_compressed(destination / "overlay.npz", timestamps=times,
        source_pos=source_pos[:, [skeleton.joint_index(x) for x in source_names]],
        source_names=np.array(source_names), source_edges=edges(source_names),
        target_pos=target_pos, target_quat_xyzw=target_quat,
        target_names=np.array(names), target_edges=edges(names),
        target_bodies=np.array([report["ik_config"]["ik_map"][x]["t_body"] for x in names]),
        orientation_bodies=np.array([report["ik_config"]["ik_map"][x]["r_body"] for x in names]))
    meta = dict(source=str(source), reference=str(reference), source_sha256=report["source_sha256"],
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        source_fps=fps, source_frames=n, output_fps=50, source_fk_max_abs_difference_m=fk_error,
        alignment="Original world coordinates; source conversion Rx(90 deg), no fitted root/yaw/height alignment",
        interpolation="Source global positions linear, target orientation SLERP at exact reference timestamps",
        raw_source="Unscaled source skeleton, metres; proportions differ from G1",
        calibrated_target="Saved SOMA scaler and native23 hand offsets, same pre-IK targets; no initialization prefix",
        metric_note="Unweighted position/orientation error against interpolated pre-IK targets; not dynamic tracking error")
    (destination / "overlay.json").write_text(json.dumps(meta, indent=2))
    print(f"Prepared {reference.parent.name}: source FK error {fk_error:.3g} m", flush=True)


def render(reference, destination, title, frame_limit=None):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from supermimic.data.motion_convert import g1_xml_path

    with np.load(reference, allow_pickle=False) as ref, np.load(destination / "overlay.npz", allow_pickle=False) as ov:
        body_names = ref["body_names"].tolist()
        root = body_names.index("pelvis")
        poses = np.c_[ref["body_pos_w"][:, root], ref["body_quat_w"][:, root], ref["joint_pos"]]
        data_overlay = {k: ov[k].copy() for k in ov.files}
        actual = ref["body_pos_w"][:, [body_names.index(x) for x in ov["target_bodies"]]]
        actual_quat = ref["body_quat_w"][:, [body_names.index(x) for x in ov["orientation_bodies"]]][:, :, [1, 2, 3, 0]]
    target = data_overlay["target_pos"]
    errors = np.linalg.norm(actual - target, axis=-1)
    orientation_errors = (Rotation.from_quat(actual_quat.reshape(-1, 4)).inv() *
        Rotation.from_quat(data_overlay["target_quat_xyzw"].reshape(-1, 4))).magnitude().reshape(errors.shape)
    metrics = {str(name): dict(mean_position_cm=float(errors[:, j].mean()*100),
        p95_position_cm=float(np.quantile(errors[:, j], .95)*100),
        max_position_cm=float(errors[:, j].max()*100),
        mean_orientation_deg=float(np.rad2deg(orientation_errors[:, j]).mean()))
        for j, name in enumerate(data_overlay["target_names"])}
    (destination / "target_errors.json").write_text(json.dumps(metrics, indent=2))
    spec = mujoco.MjSpec.from_file(str(g1_xml_path()))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[100,100,.1], rgba=[.19,.23,.29,1])
    spec.worldbody.add_light(pos=[3,-3,5], dir=[-.4,.4,-1], diffuse=[.8,.8,.8], ambient=[.4,.4,.4])
    spec.worldbody.add_light(pos=[-3,2,4], dir=[.4,-.3,-1], diffuse=[.6,.6,.6], castshadow=False)
    model = spec.compile()
    model.vis.global_.offheight = 640
    model.vis.global_.offwidth = 640
    data = mujoco.MjData(model)
    model.geom_rgba[model.geom_bodyid != 0, 3] = .42
    if model.nmat:
        model.mat_rgba[:, 3] = .42
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    cameras = []
    for azimuth, distance in [(135,3.8), (135,3.0), (45,3.0)]:
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, -13
        cameras.append(camera)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    cyan, orange, red = [.05,.95,1.,1.], [1.,.65,.08,1.], [1.,.12,.3,1.]

    def sphere(scene, point, radius, rgba):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, [radius]*3, point, np.eye(3).ravel(), rgba)
        geom.emission = .5
        scene.ngeom += 1

    def line(scene, a, b, width, rgba):
        if np.linalg.norm(b-a) < 1e-8:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        geom.emission = .5
        scene.ngeom += 1

    indices = list(range(0, len(poses), 2))
    if frame_limit:
        indices = indices[:frame_limit]
    output = destination / "overlay.mp4"
    with mujoco.Renderer(model, height=640, width=640) as renderer, \
            imageio.get_writer(output, fps=25, codec='libx264', quality=8) as writer:
        for i in indices:
            data.qpos[:] = poses[i]
            mujoco.mj_forward(model, data)
            views = []
            for panel, camera in enumerate(cameras):
                source = data_overlay['source_pos'][i]
                camera.lookat[:] = [poses[i,0], poses[i,1], .85]
                if panel == 0:
                    hip = source[data_overlay['source_names'].tolist().index('Hips')]
                    camera.lookat[:] = [.5*(poses[i,0]+hip[0]), .5*(poses[i,1]+hip[1]), 1.0]
                renderer.update_scene(data, camera=camera, scene_option=options)
                points = source if panel == 0 else target[i]
                edges = data_overlay['source_edges' if panel == 0 else 'target_edges']
                color = cyan if panel == 0 else orange
                for a, b in edges:
                    line(renderer.scene, points[a], points[b], .009, color)
                for point in points:
                    sphere(renderer.scene, point, .017, color)
                if panel:
                    for j, name in enumerate(data_overlay['target_names']):
                        line(renderer.scene, target[i,j], actual[i,j], .004, red)
                        if name in ('LeftHand','RightHand','LeftFoot','RightFoot'):
                            axes = Rotation.from_quat(data_overlay['target_quat_xyzw'][i,j]).as_matrix()
                            for axis, axiscolor in zip(range(3), ([1,0,0,1],[0,1,0,1],[.2,.5,1,1])):
                                line(renderer.scene, points[j], points[j]+axes[:,axis]*.08, .003, axiscolor)
                views.append(renderer.render().copy())
            canvas = Image.new('RGB', (1920,736), (19,23,31))
            canvas.paste(Image.fromarray(np.concatenate(views,axis=1)), (0,64))
            draw = ImageDraw.Draw(canvas)
            draw.text((15,6), f'{title} | SOMA source -> G1 native23 | t={data_overlay["timestamps"][i]:.2f}s | 1x | kinematic replay', font=font, fill='white')
            for x, label, color in [(15,'SOURCE BVH: original proportions + robot',(70,235,255)),
                    (655,'CALIBRATED IK TARGETS + robot / view A',(255,185,50)),
                    (1295,'CALIBRATED IK TARGETS + robot / view B',(255,185,50))]:
                draw.text((x,36), label, font=small, fill=color)
            hand = [data_overlay['target_names'].tolist().index(x) for x in ('LeftHand','RightHand')]
            foot = [data_overlay['target_names'].tolist().index(x) for x in ('LeftFoot','RightFoot')]
            draw.text((15,711), f'Grey: robot | Red connectors: target error | Hands L/R: {errors[i,hand[0]]*100:.1f}/{errors[i,hand[1]]*100:.1f} cm | Feet L/R: {errors[i,foot[0]]*100:.1f}/{errors[i,foot[1]]*100:.1f} cm | RGB: target XYZ axes', font=small, fill=(225,230,240))
            frame = np.asarray(canvas)
            writer.append_data(frame)
            if i == indices[len(indices)//2]:
                imageio.imwrite(destination / 'overlay.png', frame)
    meta = json.loads((destination/'overlay.json').read_text())
    meta.update(video=str(output), video_fps=25, video_frames=len(indices), reference_frames=len(poses),
                reference_duration_s=float(data_overlay['timestamps'][-1]), partial_preview=frame_limit is not None)
    (destination/'overlay.json').write_text(json.dumps(meta,indent=2))
    print(f'Rendered {title}: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, default=ROOT/'data/retargeted/soma_examples_v1')
    parser.add_argument('--reference', type=Path, help='One reference_50hz.npz, including BONES-SEED v4 clips')
    parser.add_argument('--output', type=Path, default=ROOT/'data/retargeted/soma_examples_overlay_v1')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--frame-limit', type=int, help='Short rendering probe; output marked partial')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.reference:
        jobs = [(args.reference.parent.name, args.reference.parent.name, args.reference.resolve())]
    else:
        rows = json.loads((args.examples/'summary.json').read_text())
        jobs = [(r['motion'],r['title'],(args.examples/r['motion']/'reference_50hz.npz').resolve()) for r in rows]
    if args.prepare_only:
        for slug, title, reference in jobs:
            prepare(reference, args.output/slug)
        return
    command = [str(ROOT.parent/'SOMA-Retargeter/.venv/bin/python'),str(Path(__file__).resolve()),
               '--prepare-only','--examples',str(args.examples.resolve()),'--output',str(args.output)]
    if args.reference:
        command.extend(['--reference',str(args.reference.resolve())])
    subprocess.run(command, check=True, env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    videos = [render(reference,args.output/slug,title,args.frame_limit) for slug,title,reference in jobs]
    import imageio.v2 as imageio
    with imageio.get_writer(args.output/'all_motions_overlay.mp4', fps=25, codec='libx264',quality=8) as writer:
        for video in videos:
            with imageio.get_reader(video) as reader:
                for frame in reader:
                    writer.append_data(frame)
    print(args.output/'all_motions_overlay.mp4',flush=True)


if __name__ == '__main__':
    main()
