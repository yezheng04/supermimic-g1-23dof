#!/usr/bin/env python3
"""Open NVIDIA's Robot Configurator with our URDF target selected."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "SOMA-Retargeter-official"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--screenshot", type=Path)
    a = p.parse_args()
    if a.headless:
        os.environ["PYGLET_HEADLESS"] = "1"
        if a.frames < 1:
            p.error("Headless mode requires --frames > 0")
    elif not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        p.error("Run in a graphical desktop terminal")
    import warp as wp
    from newton.viewer import ViewerGL
    from app.tools.robot_config_generator import Viewer
    wp.set_device("cuda:0")
    target = "g1_23dof_urdf_v1"

    class Native23Configurator(Viewer):
        def _refresh_robot_list(self):
            super()._refresh_robot_list()
            self.robot_list = [target] + [x for x in self.robot_list if x != target]
            self.robot_idx = 0

    viewer = ViewerGL(width=1440, height=900, headless=a.headless)
    try:
        app = Native23Configurator(viewer, json.loads((ROOT / "configs/soma_native23_configurator.json").read_text()))
        assert app.current_robot == target and len(app.current_joint_q) == 30
        assert len(app.human_offset_transforms) == 14
        viewer.set_camera(wp.vec3(2.4, -2.4, 1.5), pitch=-12, yaw=135)
        frame = 0
        while viewer.is_running() and (not a.frames or frame < a.frames):
            app.step()
            app.render()
            frame += 1
        if a.screenshot:
            from PIL import Image
            a.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(viewer.get_frame().numpy()).save(a.screenshot)
        print("Native23 configurator loaded: 30 coordinates, 14 mapped effectors")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
