"""Render clean 50fps native23 footage; all labels and controls live in the webpage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('MUJOCO_GL','egl')
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from render_world_model_demo import overlay_model


def scene():
    model=overlay_model(ghost_count=4,white_stage=True)
    model.vis.global_.offwidth=1280;model.vis.global_.offheight=900
    model.vis.quality.shadowsize=4096;model.vis.quality.offsamples=4
    model.vis.headlight.ambient[:]=[.15]*3;model.vis.headlight.diffuse[:]=[.22]*3
    model.vis.headlight.specular[:]=[.1]*3
    model.mat_rgba[model.material('silver').id]=[.82,.84,.86,1]
    model.mat_rgba[model.material('black').id]=[.055,.06,.065,1]
    model.light_ambient[model.light('key').id]=[.3]*3
    model.light_diffuse[model.light('key').id]=[.5]*3
    model.light_diffuse[model.light('fill').id]=[.23]*3
    for k in range(4):
        for g in range(model.ngeom):
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
            if name.startswith(f'future_{k}_'):
                model.geom_rgba[g]=[.08,.56,.92,[.52,.36,.22,.10][k]]
    model.vis.map.shadowclip=3
    return model


def main(a):
    report=json.loads((a.recording/'recording.json').read_text());d=np.load(a.recording/'trajectories.npz')
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    model=scene();data=mujoco.MjData(model)
    options=mujoco.MjvOption();options.geomgroup[:]=0;options.geomgroup[:3]=1;options.sitegroup[:]=0
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.elevation=-9
    renderer=mujoco.Renderer(model,height=720,width=1280)
    ff=imageio_ffmpeg.get_ffmpeg_exe();catalog=[]
    # Each mode of a motion uses the same baseline camera path, including failed branches.
    cameras={}
    for i,row in enumerate(report['runs']):
        if row['mode']!='on':continue
        n=row['reference_frames'];positions=d['states'][:n,i,:3]
        target=positions[0].copy();targets=[]
        for pos in positions:
            target=.8*target+.2*pos;targets.append(target.copy())
        quats=d['states'][:n,i,3:7]
        yaw=np.unwrap(Rotation.from_quat(quats[:,[1,2,3,0]]).as_euler('xyz')[:,2])*180/np.pi
        angles=[];angle=yaw[0]
        for value in yaw:
            angle=.97*angle+.03*value;angles.append(angle+215)
        cameras[row['slug']]=(np.array(targets),angles,d['states'][:n,i,30:33])
    def frame(i,t,kind):
        row=report['runs'][i];q=d['reference'][t,i] if kind=='reference' else d['states'][t,i,:30]
        data.qpos[:30]=q
        for k,step in enumerate(report['offset_steps']):
            offset=(k+1)*30
            visible=kind=='future' and t<row['steps'] and t+step<row['reference_frames']
            data.qpos[offset:offset+30]=d['future'][t,i,k] if visible else q
            if not visible:data.qpos[offset:offset+3]=[0,0,-50]
        targets,az,velocities=cameras[row['slug']];cam.azimuth=az[t]
        cam.lookat[:]=[targets[t,0],targets[t,1],.6]
        cam.distance=3.6 if kind=='future' else 2.65
        if kind=='future':cam.lookat[:2]+=.25*velocities[t,:2]
        if row['slug'] in ('crawl','pushup','roll'):
            cam.lookat[2]=.38;cam.distance=2.65;cam.elevation=-13
        else:cam.elevation=-9
        mujoco.mj_forward(model,data);renderer.update_scene(data,cam,options)
        # Studio contact shading avoids hard shadow-map aliasing on a white cyclorama.
        # These translucent ellipses are presentation-only, never physics geometry.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        for ring in range(12):
            scale=1.0-ring*.058
            g=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                np.array([.38*scale,.25*scale,.00015]),
                np.array([q[0],q[1],.001+ring*.0004]),np.eye(3).flatten(),
                np.array([.15,.17,.19,.022],dtype=np.float32))
            renderer.scene.ngeom+=1
        return renderer.render()
    try:
        if a.preview:
            for slug,kind in [('walk','actual'),('jog','future'),('jump','actual'),('dance','actual')]:
                i=next(i for i,r in enumerate(report['runs']) if r['slug']==slug and r['mode']=='on')
                t=min(160,report['runs'][i]['reference_frames']-1)
                Image.fromarray(frame(i,t,kind)).save(out/f'preview-{slug}-{kind}.jpg')
            return
        tasks=[]
        for i,row in enumerate(report['runs']):
            if row['mode']=='on':tasks.append((i,'actual',f"motion-{row['slug']}"))
            if row['slug'] in ('jog','jump'):tasks.append((i,'future',f"wm-{row['id']}"))
            if row['slug'] in ('flip','roll') and row['mode']=='on':tasks.append((i,'reference',f"reference-{row['slug']}"))
        for i,kind,name in tasks:
            row=report['runs'][i];n=row['reference_frames'];path=out/(name+'.mp4')
            proc=subprocess.Popen([ff,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720',
                '-r','50','-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
            start=time.monotonic()
            for t in range(n):
                rgb=frame(i,t,kind);proc.stdin.write(rgb.tobytes())
                if t==min(160,n//2):Image.fromarray(rgb).save(out/(name+'.jpg'),quality=90)
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('Encoding failed')
            catalog.append(dict(**row,kind=kind,file=name+'.mp4',poster=name+'.jpg',video_frames=n,duration=n/50))
            print(f'{name}: {n} frames, rendered in {time.monotonic()-start:.1f}s',flush=True)
        # The hero is selected excerpts of real policy execution, separated by hard cuts.
        # Each source is a newly recorded complete motion available in the gallery.
        third='kick' if (out/'motion-kick.mp4').exists() else 'crawl'
        cuts=[('dance',.6,2.4),('jump',.8,3.),(third,.3,2.4),('walk',.5,2.4)]
        parts=[]
        for k,(slug,start,length) in enumerate(cuts):
            part=out/f'.hero-{k}.mp4';parts.append(part)
            subprocess.run([ff,'-y','-loglevel','error','-ss',str(start),'-i',str(out/f'motion-{slug}.mp4'),
                '-t',str(length),'-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(part)],check=True)
        listing=out/'.hero-concat.txt';listing.write_text(''.join(f"file '{p.resolve()}'\n" for p in parts))
        subprocess.run([ff,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-c','copy',
            '-movflags','+faststart',str(out/'hero.mp4')],check=True)
        subprocess.run([ff,'-y','-loglevel','error','-ss','1','-i',str(out/'hero.mp4'),'-frames:v','1',str(out/'hero.jpg')],check=True)
        for p in parts+[listing]:p.unlink()
        (out/'catalog.json').write_text(json.dumps(dict(recording=report,clips=catalog,hero_cuts=cuts,camera={'view':'front-quarter','heading_offset_degrees':215,'position_smoothing':.8,'future_lookahead_seconds':.25}),ensure_ascii=False,indent=2)+'\n')
    finally:renderer.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview',action='store_true');main(p.parse_args())
