"""Export native23 physics, visuals and checkpoint-specific latent-conditioned actors."""
import argparse,json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np,torch
from supermimic.data.packed_library import load_pack,sha256
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.envs import make_env,SuperMimicEnvCfgArgs
from supermimic.envs.state_io import StateIO
from supermimic.world_model.heads import analytic_observation,flatten_history_term_major

@torch.no_grad()
def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    selected=json.loads(a.selection.read_text())['selected'];source=load_pack(a.library)
    indices={n:i for i,n in enumerate(source['names'])};chunks=[];lengths=[];starts=[]
    for r in selected:
        i=indices[r['name']];n=int(source['lengths'][i]);st=int(source['starts'][i]);starts.append(sum(lengths));lengths.append(n);chunks.append(source['ref'][st:st+n])
    pack={**source,'ref':torch.cat(chunks),'starts':starts,'lengths':lengths,'names':[r['name'] for r in selected],'manifest_sha256':None}
    subset=out.parent/'browser_motions.pt';torch.save(pack,subset);del pack,source,chunks
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(subset)],num_envs=1,episode_length_s=1e5,action_noise_std=0.,obs_noise=False,reset_noise=False,domain_randomization=False,sampling_mode='start',record_transitions=False,compact_library=True,paper_protocol=True),device='cuda:0')
    try:
        cmd=env.command_manager.get_term('motion');lib=cmd.library;cmd.forced_frame[:]=0;obs,_=env.reset()
        model=env.sim.mj_model;term=env.action_manager.get_term('joint_pos');q0=env.scene['robot'].data.default_joint_pos[0]
        # Preserve actual collision material values after deterministic reset.
        model.geom_friction[:]=env.sim.model.geom_friction[0].cpu().numpy()
        xml=out/'physics.xml';spec=env.scene.spec.copy()
        for key in ('timestep','iterations','ls_iterations','integrator','solver','cone','jacobian','tolerance','ls_tolerance','impratio','disableflags','enableflags'):
            setattr(spec.option,key,getattr(model.opt,key))
        for geom in spec.geoms:
            gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,geom.name)
            if gid>=0:geom.friction[:]=model.geom_friction[gid]
        xml.write_text(spec.to_xml())
        tree=ET.parse(xml);root=tree.getroot()
        # Visual meshes have zero density/contact; physics keeps all native collision primitives.
        # Compiled inertials are explicit, so removing visual-only geoms does not change mass.
        for parent in root.iter():
            for child in list(parent):
                if child.tag=='geom' and (child.get('type')=='mesh' or child.get('mesh')):
                    assert child.get('class','').endswith('visual') or (child.get('contype','1')=='0' and child.get('conaffinity','1')=='0')
                    parent.remove(child)
                elif child.tag in ('mesh','texture','material'):parent.remove(child)
                else:
                    child.attrib.pop('material',None)
        root.find('compiler').attrib.pop('meshdir',None)
        from mjlab.utils.xml import fix_spec_xml
        xml.write_text(fix_spec_xml(ET.tostring(root,encoding='unicode')))
        physics=mujoco.MjModel.from_xml_path(str(xml))
        bodymap={i:mujoco.mj_name2id(physics,mujoco.mjtObj.mjOBJ_BODY,model.body(i).name) for i in range(model.nbody)}
        for i,j in bodymap.items():
            if j>=0:
                np.testing.assert_allclose(model.body_mass[i],physics.body_mass[j],atol=2e-5)
                np.testing.assert_allclose(model.body_inertia[i],physics.body_inertia[j],atol=2e-5)
            else:assert model.body_mass[i]==0
        for key in ('dof_armature','dof_damping','actuator_gainprm','actuator_biasprm','actuator_forcerange','jnt_range'):
            np.testing.assert_allclose(getattr(model,key),getattr(physics,key),atol=2e-5,rtol=2e-5)
        assert (physics.nq,physics.nv,physics.nu)==(model.nq,model.nv,model.nu)
        meshes=[];geoms=[];binary=bytearray()
        for i in range(model.nmesh):
            va=int(model.mesh_vertadr[i]);vn=int(model.mesh_vertnum[i]);fa=int(model.mesh_faceadr[i]);fn=int(model.mesh_facenum[i])
            vertices=np.asarray(model.mesh_vert[va:va+vn],dtype='<f4');faces=np.asarray(model.mesh_face[fa:fa+fn],dtype='<u4')
            vo=len(binary);binary.extend(vertices.tobytes());fo=len(binary);binary.extend(faces.tobytes());meshes.append({'vertices':[vo,vn*3],'indices':[fo,fn*3]})
        for g in range(model.ngeom):
            if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mid=int(model.geom_matid[g]);color=model.mat_rgba[mid] if mid>=0 else model.geom_rgba[g]
            geoms.append({'body':bodymap[int(model.geom_bodyid[g])],'mesh':int(model.geom_dataid[g]),'pos':model.geom_pos[g].tolist(),'quat':model.geom_quat[g].tolist(),'color':color.tolist()})
        (out/'robot.bin').write_bytes(binary);(out/'robot.json').write_text(json.dumps({'meshes':meshes,'geoms':geoms,'nbody':int(physics.nbody)}))
        def vector(v):return v[0].cpu().tolist() if isinstance(v,torch.Tensor) else [v]*23
        target_ids=term.target_ids.cpu().tolist();actmap=[target_ids.index(int(model.jnt_qposadr[j])-7) for j in model.actuator_trnid[:,0]]
        control={'q0':q0.cpu().tolist(),'scale':vector(term.scale),'offset':vector(term.offset),'actuator_action_index':actmap,'dt':.02,'substeps':4,'physics_dt':.005,'history':10,'obs_dim':750,'latent_dim':32}
        initial=[];motiondata=[]
        labels={'walk':'向前行走','frog':'蛙跳','jog':'慢跑','dance':'街舞','kick':'踢腿','punch':'出拳'}
        for i,r in enumerate(selected):
            st=int(lib.clip_start[i]);n=lengths[i];cmd.forced_frame[:]=st;env.reset();state=StateIO(env).read()[0].cpu().numpy();initial.append(state.tolist())
            ids=torch.arange(st,st+n,device='cuda:0');rp,rvel=lib.ref_qpos_qvel(ids)
            motiondata.append({'id':r['slug'],'category':r['category'],'title':labels[r['category']]+(' · 2' if r['slug'].endswith('2') else ''),'name':r['name'],'frames':n,'initial':state.tolist(),'height':rp[:,2].cpu().tolist(),'gravity_z':(2*(rp[:,4]**2+rp[:,5]**2)-1).cpu().tolist(),'screening_80k':r['screening']['norm_return']})
        policies=[];fixtures=[]
        for ck in a.checkpoint:
            policy,cfg=load_policy_from_checkpoint(ck,'cpu');iteration=torch.load(ck,map_location='cpu',weights_only=False)['iteration'];pid=str(iteration)
            class Actor(torch.nn.Module):
                def __init__(self,p):super().__init__();self.decoder=p.decoder
                def forward(self,history,latent):return self.decoder(latent,history)
            actor=Actor(policy).eval();h=torch.zeros(1,750);z=torch.zeros(1,32)
            torch.onnx.export(actor,(h,z),str(out/f'actor-{pid}.onnx'),input_names=['history','latent'],output_names=['action'],opset_version=17,dynamo=False)
            embeddings=[]
            for i,r in enumerate(selected):
                st=int(lib.clip_start[i]);n=lengths[i];frames=torch.arange(st,st+n,device='cuda:0');goals=lib.goal_window(frames,10).cpu()
                embeddings.append(policy.encoders['motion'](goals).numpy().astype('<f4'))
                if i==0:
                    state=torch.tensor(initial[i])[None];o=analytic_observation(state,torch.zeros(1,23),torch.zeros(1,27),q0.cpu());h=flatten_history_term_major(o.repeat(10,1,1));z=torch.from_numpy(embeddings[-1][:1]);pred=policy(h,goals[:1])
                    assert torch.allclose(pred,actor(h,z),atol=1e-6)
                    fixtures.append({'policy':pid,'history':h.flatten().tolist(),'latent':z.flatten().tolist(),'action':pred.flatten().tolist()})
            np.concatenate(embeddings).tofile(out/f'latents-{pid}.bin')
            policies.append({'id':pid,'label':f'{iteration//1000}k','file':f'actor-{pid}.onnx','latents':f'latents-{pid}.bin','source_sha256':sha256(ck)})
        frame=0
        for r in motiondata:r['offset']=frame;frame+=r['frames']
        result={'schema':1,'control':control,'motions':motiondata,'policies':policies,'native_physics':'3.5.0','browser_physics':'3.8.0','goal_encoding':'Checkpoint-specific 32D encoder outputs; decoder runs live on simulated observations'}
        (out/'catalog.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
        (out.parent/'inference_fixtures.json').write_text(json.dumps(fixtures))
        # Native CPU oracle: same stripped physics and live actor, for WASM parity checking.
        reference=[];policy,_=load_policy_from_checkpoint(a.checkpoint[0],'cpu');dd=mujoco.MjData(physics)
        dd.qpos[:]=initial[0][:30];dd.qvel[:]=initial[0][30:];mujoco.mj_forward(physics,dd);prev=torch.zeros(1,23);hist=None
        for t in range(60):
            s=torch.tensor(np.concatenate([dd.qpos,dd.qvel]),dtype=torch.float32)[None];o=analytic_observation(s,prev,torch.zeros(1,27),q0.cpu())
            hist=o.repeat(10,1,1) if hist is None else torch.cat([hist[1:],o[None]])
            action=policy(flatten_history_term_major(hist),lib.goal_window(torch.tensor([t],device='cuda:0'),10).cpu());prev=action
            target=(action[0].numpy()*np.array(control['scale'])+np.array(control['offset']))
            dd.ctrl[:]=target[actmap]
            for _ in range(4):mujoco.mj_step(physics,dd)
            reference.append({'qpos':dd.qpos.tolist(),'action':action[0].tolist()})
        (out.parent/'native_browser_oracle.json').write_text(json.dumps(reference))
        print('Exported',len(motiondata),'motions',len(policies),'actors',sum(p.stat().st_size for p in out.iterdir())/1e6,'MB',flush=True)
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--checkpoint',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
