"""Single white-stage view with fading future poses, inspired by the official viewer."""
import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL","egl")

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from render_world_model_demo import overlay_model
from supermimic.robots.spec import NQ


def main(args):
    folder=args.folder
    info=json.loads((folder/"prediction.json").read_text())
    source=Path(info["source"])
    d=np.load(source/"trajectories.npz",allow_pickle=False)
    future=np.load(folder/"future_trails.npz",allow_pickle=False)
    recording=json.loads((source/"recording.json").read_text())
    model=overlay_model(ghost_count=8,white_stage=True)
    model.vis.global_.offwidth=1600;model.vis.global_.offheight=900
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=900,width=1600)
    opt=mujoco.MjvOption();opt.geomgroup[:]=0;opt.geomgroup[:3]=1;opt.sitegroup[:]=0
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth=140;camera.elevation=-12
    fonts={n:ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',n) for n in (18,22,26,34)}
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    (folder/'clips').mkdir(exist_ok=True);(folder/'previews').mkdir(exist_ok=True)
    outputs=[]
    try:
        for i,row in enumerate(recording['clips']):
            ids=np.flatnonzero(future['entries'][:,0]==i)
            if args.preview_only: ids=ids[len(ids)//2:len(ids)//2+1]
            out=folder/'clips'/f"{i+1:02d}_{row['name']}.mp4"
            ff=None
            if not args.preview_only:
                ff=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1600x900',
                    '-r','25','-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
            for j,index in enumerate(ids):
                t=int(future['entries'][index,1]); count=int(future['visible_counts'][index])
                current=d['states'][t,i,:NQ];pred=future['qpos'][index]
                data.qpos[:NQ]=current
                for k in range(8):
                    data.qpos[(k+1)*NQ:(k+2)*NQ]=pred[k] if k<count else current
                    if k>=count:data.qpos[(k+1)*NQ:(k+1)*NQ+3]=[0,0,-50]
                # Camera framing moves only the viewpoint, never either trajectory.
                endpoint=pred[count-1,:3] if count else current[:3]
                center=.7*current[:3]+.3*endpoint
                camera.lookat[:]=[center[0],center[1],.5 if current[2]<.5 else .7]
                camera.distance=max(2.6,2.4+float(np.linalg.norm(endpoint-current[:3]))*.7)
                mujoco.mj_forward(model,data)
                renderer.update_scene(data,camera,opt)
                image=Image.fromarray(renderer.render())
                draw=ImageDraw.Draw(image)
                draw.text((40,26),row['title'],font=fonts[34],fill=(25,30,38))
                draw.text((40,76),'G1 23DoF  /  Checkpoint 80,000',font=fonts[22],fill=(110,116,125))
                draw.ellipse((1112,35,1128,51),fill=(105,110,116));draw.text((1140,24),'仿真执行',font=fonts[26],fill=(55,60,68))
                draw.ellipse((1320,35,1336,51),fill=(30,155,240));draw.text((1348,24),'世界模型',font=fonts[26],fill=(55,60,68))
                draw.text((40,791),'Policy on  ·  策略在预测状态上递推  ·  蓝色虚影：未来 0.08–0.64 秒',font=fonts[22],fill=(80,90,104))
                draw.line((40,840,1560,840),fill=(222,227,233),width=3)
                draw.line((40,840,40+1520*t/max(row['steps'],1),840),fill=(42,146,225),width=3)
                draw.text((40,855),f"{t/50:.2f} s / {row['seconds']:.2f} s    ·    Frame {t}/{row['steps']}    ·    1×",font=fonts[18],fill=(100,108,119))
                if row['failed']:
                    draw.text((1130,855),'此片段触发失败终止',font=fonts[18],fill=(190,70,55))
                if args.preview_only or j==len(ids)//2:image.save(folder/'previews'/f'{i+1:02d}.png')
                if ff:ff.stdin.write(np.asarray(image).tobytes())
            if ff:
                ff.stdin.close()
                if ff.wait()!=0:raise RuntimeError('Encoding failed')
                outputs.append(out)
                print(f'Rendered {i+1}/{len(recording["clips"])}: {row["title"]}',flush=True)
        if outputs:
            listing=folder/'concat.txt';listing.write_text(''.join(f"file '{p.resolve()}'\n" for p in outputs))
            final=folder/'g1_23dof_80k_future_trail.mp4'
            subprocess.run([ffmpeg,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(final)],check=True)
            (folder/'video.json').write_text(json.dumps(dict(path=str(final),fps=25,width=1600,height=900,frames=len(future['entries']),
                seconds=len(future['entries'])/25,mode='Policy on, rolling future trajectory trail; not simultaneous state error'),indent=2)+'\n')
            print(final,flush=True)
    finally:renderer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=Path('reports/demo_80k_future_trail_20261030'))
    p.add_argument('--preview-only',action='store_true')
    main(p.parse_args())
