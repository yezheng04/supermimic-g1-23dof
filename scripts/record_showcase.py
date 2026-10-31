"""Fresh 50 Hz simulation recordings and 32-step actor-in-model trails for the website."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from record_world_model_demo import TerminalCapture
from supermimic.checkpoints import load_checkpoint
from supermimic.cli import apply_config
from supermimic.config import TrainConfig
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.packed_library import load_pack, sha256, validate_pack
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.sim_context import SimContext
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import FOOT_GEOM_REGEX, ROOT_BODY, TRACKED_BODIES
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major
from supermimic.world_model.world_model import WorldModel

CLIPS = [
    ('walk', '行走', '230125__walk_ff_loop_315_001__A146'),
    ('jog', '慢跑', '230906__jog_ff_loop_180_R_slow_002__A448'),
    ('jump', '跳跃', '230524__high_jump_R_001__A395'),
    ('dance', '街舞', '230413__dance_hiphop_prep_R_fast_002__A317'),
    ('pushup', '俯卧撑', '230509__push_up_loop_002__A359'),
    ('crawl', '爬行', '230120__crawl_ff_loop_270_001__A133'),
    ('flip', '360° 翻转', '230531__flip_360_001__A415'),
    ('roll', '侧滚', '230531__side_roll_R_004__A415'),
]


@torch.no_grad()
def future_rollout(wm, policy, lib, state, hist, xi, frames, q0, hold, held_action, horizon=32):
    """Future external events are not revealed; use only context known at the seed."""
    result, first = [], None
    last = lib.last_frame_of(frames)
    for k in range(horizon):
        action = policy(flatten_history_term_major(hist), lib.goal_window(torch.minimum(frames+k, last), 10))
        action = torch.where(hold[:, None], held_action, action)
        if k == 0:
            first = action.clone()
        state = wm.step(state, action, xi)
        obs = analytic_observation(state, action, xi, q0)
        hist = torch.cat([hist[1:], obs[None]])
        if (k+1) % 8 == 0:
            result.append(state[:, :30].clone())
    return torch.stack(result, 1), first


@torch.no_grad()
def main(a):
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Use a new recording directory')
    a.output.mkdir(parents=True, exist_ok=True)
    clips = CLIPS
    if a.selection:
        labels={'walk':'向前行走','jog':'慢跑','frog':'蛙跳','dance':'街舞','kick':'踢腿','punch':'出拳'}
        rows=json.loads(a.selection.read_text())['selected']
        rows=[r for r in rows if r['slug']==r['category']]
        clips=[('jump' if r['slug']=='frog' else r['slug'],labels[r['category']],r['name']) for r in rows]
    ck = load_checkpoint(a.checkpoint)
    assert ck['iteration'] == 80000
    cfg = apply_config(TrainConfig(), ck['config'])
    source = load_pack(a.library)
    indices = {name:i for i,name in enumerate(source['names'])}
    chunks, lengths, starts = [], [], []
    for _, _, name in clips:
        i = indices[name]; n = int(source['lengths'][i]); start = int(source['starts'][i])
        starts.append(sum(lengths)); lengths.append(n); chunks.append(source['ref'][start:start+n])
    pack = {**source, 'names':[c[2] for c in clips], 'starts':starts, 'lengths':lengths,
            'ref':torch.cat(chunks), 'manifest_sha256':None}
    validate_pack(pack); torch.save(pack, a.output/'motions.pt')
    del pack, source, chunks
    runs = [(i, 'on') for i in range(len(clips))]+[(i,m) for i,c in enumerate(clips) if c[0] in ('jog','jump') for m in ('hold','slip')]
    env = make_env(SuperMimicEnvCfgArgs(npz_files=[str(a.output/'motions.pt')], num_envs=len(runs),
        episode_length_s=1e5, action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode='start', record_transitions=False,
        seed=0, compact_library=True, paper_protocol=True), device='cuda:0')
    try:
        robot=env.scene['robot']; io=StateIO(env); context=SimContext(env,FOOT_GEOM_REGEX)
        capture=TerminalCapture(io);env.recorder_manager.terms['showcase_terminal']=capture
        cmd=env.command_manager.get_term('motion');lib=cmd.library
        clip_ids=torch.tensor([i for i,m in runs],device='cuda:0')
        start_frames=lib.clip_start[clip_ids]
        cmd.forced_frame[:]=start_frames
        obs,_=env.reset()
        policy=Policy(cfg.policy).cuda();policy.load_state_dict(ck['policy']);policy.eval()
        q0=robot.data.default_joint_pos[0].clone()
        heads=AnalyticHeads(G1Kinematics(load_g1_model(),device='cuda:0'),TRACKED_BODIES,ROOT_BODY,ROOT_BODY,
                           robot.data.soft_joint_pos_limits[0].clone(),q0)
        wm=WorldModel(cfg.world_model,heads,'cuda:0');wm.load_state_dict(ck['wm']);wm.eval()
        hold_mask=torch.tensor([m=='hold' for i,m in runs],device='cuda:0')
        slip_ids=torch.tensor([k for k,(_,m) in enumerate(runs) if m=='slip'],device='cuda:0')
        local_geoms,_=robot.find_geoms(FOOT_GEOM_REGEX)
        geom_ids=robot.indexing.geom_ids[local_geoms]
        friction=env.sim.model.geom_friction
        assert friction.ndim==3 and friction.shape[0]==len(runs), friction.shape
        event_frame=100
        states=[io.read()]; actions=[]; contexts=[]
        n=len(runs);alive=torch.ones(n,dtype=torch.bool,device='cuda:0')
        steps=torch.zeros(n,dtype=torch.long,device='cuda:0');failed=torch.zeros_like(alive)
        reasons=[None]*n;returns=torch.zeros(n,device='cuda:0');held=torch.zeros(n,23,device='cuda:0')
        max_steps=max(lengths)-1
        for t in range(max_steps):
            if t==event_frame:
                # Branch from the same actual rollout, not independent trajectories
                # whose contact numerics may already have drifted before intervention.
                branch_ids=torch.tensor([k for k,(_,m) in enumerate(runs) if m!='on'],device='cuda:0')
                baseline_ids=torch.tensor([runs[k][0] for k in branch_ids.tolist()],device='cuda:0')
                for value in states+actions+contexts:
                    value[branch_ids]=value[baseline_ids].clone()
                warmstart=env.sim.data.qacc_warmstart[baseline_ids].clone()
                io.write(branch_ids,states[-1][baseline_ids])
                env.sim.data.qacc_warmstart[branch_ids]=warmstart
                for buffers in env.observation_manager._group_obs_term_history_buffer.values():
                    for buffer in buffers.values():
                        if buffer._buffer is not None:
                            buffer._buffer[:,branch_ids]=buffer._buffer[:,baseline_ids].clone()
                for value in obs.values():
                    value[branch_ids]=value[baseline_ids].clone()
                for value in (env.action_manager.action,env.action_manager.prev_action,
                              env.action_manager.prev_prev_action):
                    value[branch_ids]=value[baseline_ids].clone()
                returns[branch_ids]=returns[baseline_ids].clone()
                steps[branch_ids]=steps[baseline_ids].clone()
                held=actions[-1].clone()
                friction[slip_ids[:,None],geom_ids[None,:],0]=.3
            xi=context.read();contexts.append(xi)
            expected=torch.ones(n,device='cuda:0')
            if t>=event_frame:expected[slip_ids]=.3
            assert torch.allclose(xi[:,0],expected,atol=1e-6), 'Friction/context mismatch'
            action=policy(obs['policy'],obs['goal'])
            if t>=event_frame:action=torch.where(hold_mask[:,None],held,action)
            actions.append(action.clone())
            obs,reward,term,timeout,_=env.step(action)
            reached=io.read();done=term|timeout
            reached[done]=capture.states[done];reached[~alive]=states[-1][~alive]
            states.append(reached);returns+=reward*alive;steps+=alive.long()
            for k in (done&alive).nonzero().flatten().tolist():
                failed[k]=term[k];reasons[k]='+'.join(key for key,value in env.failure_masks.items() if bool(value[k])) or None
            alive &= ~done
        states=torch.stack(states);actions=torch.stack(actions);contexts=torch.stack(contexts)
        previous=torch.cat([torch.zeros_like(actions[:1]),actions])
        observations=analytic_observation(states,previous,torch.cat([contexts,contexts[-1:]]),q0)
        entries=[(k,t) for k in range(n) for t in range(int(steps[k]))]
        trails=np.zeros((max_steps+1,n,4,30),dtype=np.float32)
        error=0.
        for offset in range(0,len(entries),256):
            pairs=entries[offset:offset+256]
            ids=torch.tensor([k for k,t in pairs],device='cuda:0');times=torch.tensor([t for k,t in pairs],device='cuda:0')
            hist_times=(times[None]+torch.arange(-9,1,device='cuda:0')[:,None]).clamp_min(0)
            hist=observations[hist_times,ids[None]]
            predicted,first=future_rollout(wm,policy,lib,states[times,ids],hist,contexts[times,ids],
                start_frames[ids]+times,q0,hold_mask[ids]&(times>=event_frame),held[ids])
            error=max(error,float((first-actions[times,ids]).abs().max()))
            assert torch.isfinite(predicted).all()
            trails[times.cpu().numpy(),ids.cpu().numpy()]=predicted.cpu().numpy()
            print(f'Predicted {min(offset+256,len(entries))}/{len(entries)} frames',flush=True)
        assert error<.01, ('History/action reconstruction error',error)
        reference=np.zeros((max_steps+1,n,30),dtype=np.float32)
        for t in range(max_steps+1):
            frames=torch.minimum(start_frames+t,lib.last_frame_of(start_frames))
            reference[t]=lib.ref_qpos_qvel(frames)[0].cpu().numpy()
        np.savez_compressed(a.output/'trajectories.npz',states=states.cpu().numpy(),actions=actions.cpu().numpy(),
            contexts=contexts.cpu().numpy(),future=trails,reference=reference,steps=steps.cpu().numpy())
        records=[]
        for k,(i,mode) in enumerate(runs):
            slug,title,name=clips[i]
            records.append(dict(id=f'{slug}-{mode}',slug=slug,title=title,name=name,mode=mode,
                reference_frames=lengths[i],steps=int(steps[k]),failed=bool(failed[k]),failure_reason=reasons[k],
                return_full=float(returns[k])/(.1*(lengths[i]-1)),event_frame=event_frame if mode!='on' else None))
        metadata=dict(recorded_on='2026-11-01',checkpoint_iteration=80000,checkpoint_sha256=sha256(a.checkpoint),
            fps=50,horizon=32,offset_steps=[8,16,24,32],first_action_max_error=error,
            reset='First reference frame, zero noise, no domain randomization',
            modes={'on':'Policy acts on current or imagined observations',
                   'hold':'At frame 100 retain the previous action; PD remains active, not torque-off',
                   'slip':'At frame 100 foot friction becomes 0.3, model receives measured current context'},
            prediction='Reseed each frame; no real future states/actions/context; active external event only',
            branching='Shared baseline prefix through frame 100; copy qpos/qvel, solver warmstart, observation and action histories at intervention',
            termination='Freeze captured pre-reset state after first failure/clip end; hide future ghosts',runs=records)
        (a.output/'recording.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(records,ensure_ascii=False,indent=2),flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection',type=Path)
    main(p.parse_args())
