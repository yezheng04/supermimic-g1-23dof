"""Isolated fixed-state bootstrap/gradient diagnostics; never writes training weights."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.checkpoints import load_checkpoint
from supermimic.cli import apply_config
from supermimic.config import TrainConfig
from supermimic.data.motion_convert import load_g1_model
from supermimic.data.packed_library import load_pack, sha256, validate_pack
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.tracking_env_cfg import ROOT_BODY, TRACKED_BODIES
from supermimic.imagination.objective import gae_targets, policy_objective
from supermimic.imagination.rollout import imagine
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy
from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major
from supermimic.world_model.world_model import WorldModel


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare(out):
    sources = [Path("reports/demo_80k_world_model_20261030"), Path("reports/demo_80k_flips_20261030")]
    original = load_pack("reports/evaluation_100k_20261030/library.pt")
    names, cases, arrays = [], [], []
    for folder in sources:
        d = np.load(folder / "trajectories.npz")
        for i, name in enumerate(d["names"].tolist()):
            names.append(name)
            steps = int(d["steps"][i])
            times = [20, min(60, steps-1)]
            if i == 0 and folder == sources[0]:
                assert not bool(d["failed"][i])
                times += [int(d["reference_frames"][i])-6, int(d["reference_frames"][i])-2]
            for t in times:
                assert 10 <= t < steps
                hs = d["states"][t-9:t+1, i]
                ha = d["actions"][t-10:t, i]
                arrays.append(dict(state=d["states"][t,i], xi=d["contexts"][t,i], hist_states=hs, hist_actions=ha))
                cases.append(dict(name=name, local_frame=t, source=str(folder),
                                  source_sha256=sha256(folder / "trajectories.npz")))
    chunks, starts, lengths = [], [], []
    for name in names:
        i=original["names"].index(name);s,n=int(original["starts"][i]),int(original["lengths"][i])
        starts.append(sum(lengths));lengths.append(n);chunks.append(original["ref"][s:s+n])
    pack={**original,"names":names,"starts":starts,"lengths":lengths,"ref":torch.cat(chunks),"manifest_sha256":None}
    validate_pack(pack);torch.save(pack,out/"library.pt")
    batch={key:torch.from_numpy(np.stack([r[key] for r in arrays],axis=1 if key.startswith('hist_') else 0))
           for key in arrays[0]}
    batch["frame"]=torch.tensor([starts[names.index(c["name"])]+c["local_frame"] for c in cases])
    batch["prev_action"]=batch["hist_actions"][-1].clone()
    torch.save(batch,out/"fixed_starts.pt")
    save(out/"fixed_starts.json",dict(cases=cases,library_sha256=sha256(out/"library.pt"),
        starts_sha256=sha256(out/"fixed_starts.pt"),
        caveat="24 fixed actual states/histories recorded under the 80k policy, no noise/DR. Cross-checkpoint inputs are controlled but off-policy; not the original replay batch or an exact replay of the instability event."))


def stats(x):
    x=x.detach().float().flatten()
    if not torch.isfinite(x).all():raise ValueError("Nonfinite diagnostic tensor")
    if not x.numel():return None
    return dict(mean=float(x.mean()), min=float(x.min()), max=float(x.max()),
                median=float(x.median()), fraction_above_10=float((x>10).float().mean()))


def flat_gradient(objective, parameters):
    grads=torch.autograd.grad(objective,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,grads)]).detach()


def cosine(a,b):
    return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20))


def assign_gradient(parameters, gradient):
    offset=0
    for p in parameters:
        p.grad=gradient[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()


def main(args):
    out=args.output
    torch.manual_seed(20261030)
    if not (out/"fixed_starts.json").exists():prepare(out)
    protocol=json.loads((out/"fixed_starts.json").read_text())
    assert sha256(out/"library.pt")==protocol["library_sha256"]
    assert sha256(out/"fixed_starts.pt")==protocol["starts_sha256"]
    manifest=json.loads((out/"manifest.json").read_text())
    checkpoints={}
    for iteration in {args.iteration,args.critic_iteration or args.iteration}:
        m=next(r for r in manifest if r["iteration"]==iteration);p=out/m["path"]
        assert sha256(p)==m["sha256"]
        checkpoints[iteration]=load_checkpoint(p)
    ck=checkpoints[args.iteration];critic_it=args.critic_iteration or args.iteration;cck=checkpoints[critic_it]
    cfg=apply_config(TrainConfig(),ck["config"])
    label=f"actor_wm_{args.iteration:07d}_critic_{critic_it:07d}"
    if (out/f"{label}.json").exists():raise FileExistsError(label)
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(out/"library.pt")],num_envs=1,
        record_transitions=False,compact_library=True,paper_protocol=True,obs_noise=False,
        action_noise_std=0,reset_noise=False,domain_randomization=False,seed=0),device="cuda:0")
    try:
        precision_before=dict(mode=torch.get_float32_matmul_precision(),tf32=torch.backends.cuda.matmul.allow_tf32)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        print('Precision before audit override:',precision_before,flush=True)
        robot=env.scene["robot"];lib=env.command_manager.get_term("motion").library
        heads=AnalyticHeads(G1Kinematics(load_g1_model(),device="cuda:0"),TRACKED_BODIES,ROOT_BODY,ROOT_BODY,
            robot.data.soft_joint_pos_limits[0].clone(),robot.data.default_joint_pos[0].clone())
        wm=WorldModel(cfg.world_model,heads,"cuda:0");wm.load_state_dict(ck["wm"]);wm.requires_grad_(False)
        policy=Policy(cfg.policy).cuda();policy.load_state_dict(ck["policy"])
        critic=Critic(cfg.critic,pcfg=cfg.policy).cuda();critic.load_state_dict(cck["critic"]);critic.requires_grad_(False)
        raw=torch.load(out/"fixed_starts.pt",weights_only=True)
        start=StartBatch(**{k:v.cuda() for k,v in raw.items()})
        H=cfg.policy_trainer.horizon;gamma=cfg.policy_trainer.gamma;lam=cfg.policy_trainer.gae_lambda
        ro=imagine(wm,policy,lib,start,H,obs_noise=False,anchor_obs=False)
        H1,n,_=ro.states.shape
        feat=wm.spec.canonical(ro.states.reshape(H1*n,-1)).reshape(H1,n,-1)
        xi=wm.spec.normalize_xi(start.xi)[None].expand(H1,n,-1)
        values=critic(feat.reshape(H1*n,-1),ro.goals.reshape(H1*n,-1),xi.reshape(H1*n,-1),ro.e_root.reshape(H1*n,-1)).reshape(H1,n)
        valid=ro.frames[:-1]<lib.last_frame_of(ro.frames[0])[None]
        targets=gae_targets(ro.rewards,ro.dhat[1:],values,gamma,lam)
        bounded_targets=gae_targets(ro.rewards,ro.dhat[1:],values.clamp_max(10),gamma,lam)
        assert not targets.requires_grad and float(bounded_targets.max())<=10.0001
        reward_part=policy_objective(ro,None,gamma)
        bootstrap=((gamma**H)*ro.survival[-1]*values[-1]).mean()
        value_path=((gamma**H)*ro.survival[-1].detach()*values[-1]).mean()
        survival_path=((gamma**H)*ro.survival[-1]*values[-1].detach()).mean()
        parameters=list(policy.parameters())
        gradients={k:flat_gradient(-v/H,parameters) for k,v in
                   [("reward",reward_part),("value_state",value_path),("survival",survival_path)]}
        full=flat_gradient(-policy_objective(ro,values[-1],gamma)/H,parameters)
        decomposed=sum(gradients.values())
        rel_error=float((full-decomposed).norm()/full.norm().clamp_min(1e-12))
        print('Gradient identity relative error:',rel_error,'norms:',{k:float(v.norm()) for k,v in gradients.items()},'full:',float(full.norm()),flush=True)
        assert rel_error<1e-3 and all(p.grad is None for p in critic.parameters()) and all(p.grad is None for p in wm.parameters())
        state_grad=torch.autograd.grad(values[-1].sum(),ro.states,retain_graph=True)[0][-1]
        boundary=(lib.last_frame_of(start.frame)-start.frame)<H
        assert bool((ro.survival[-1,boundary]==0).all())
        terminal=ro.dhat[1:]==1
        torch.testing.assert_close(targets[terminal],ro.rewards.detach()[terminal],atol=1e-6,rtol=1e-5)
        print(label,"rollout and gradient split complete",flush=True)
        report=dict(iteration=args.iteration,critic_iteration=critic_it,fixed_starts_sha256=sha256(out/"fixed_starts.pt"),
            horizon=H,batch_size=n,obs_noise=False,compiled=False,matmul_precision='highest',tf32=False,
            objective=dict(reward=float(reward_part.detach()),bootstrap=float(bootstrap.detach()),total=float((reward_part+bootstrap).detach())),
            values_at_actual_starts=stats(values[0]),values_at_imagined_end=stats(values[-1]),
            values_at_valid_training_states=stats(values[:-1][valid]),targets=stats(targets[valid]),
            targets_with_bootstrap_capped_at_10_DIAGNOSTIC_ONLY=stats(bounded_targets[valid]),
            survival_H=stats(ro.survival[-1]),rewards=stats(ro.rewards[valid]),
            gradient_norms={k:float(v.norm()) for k,v in gradients.items()},full_gradient_norm=float(full.norm()),
            value_to_reward_gradient_norm_ratio=float(gradients["value_state"].norm()/gradients["reward"].norm().clamp_min(1e-12)),
            reward_value_gradient_cosine=cosine(gradients["reward"],gradients["value_state"]),
            gradient_decomposition_relative_error=rel_error,
            critic_state_gradient_norm=stats(state_grad.norm(dim=-1)),
            checks=dict(frozen_wm_critic_no_parameter_grad=True,targets_detached=True,
                terminal_target_equals_terminal_reward=True,clip_end_bootstrap_zero=True,bounded_targets_with_bounded_values=True),
            actual_state_cases=[dict(**c,value=float(values[0,i].detach()),end_value=float(values[-1,i].detach()),
                                     survival=float(ro.survival[-1,i].detach())) for i,c in enumerate(protocol['cases'])])
        # Counterfactual single Adam steps use identical saved optimizer momentum.
        # They measure action change only; they are not proposed fixes or training performance tests.
        obs=flatten_history_term_major(torch.stack([analytic_observation(start.hist_states[i],start.hist_actions[i],start.xi,lib.default_joint_pos) for i in range(10)]))
        goal=lib.goal_window(start.frame,10)
        with torch.no_grad():initial_action=policy(obs,goal).clone()
        report['actor_one_step_counterfactuals']={}
        for key,g in [('full',full),('detach_value_state',gradients['reward']+gradients['survival']),('reward_only',gradients['reward'])]:
            policy.load_state_dict(ck['policy'])
            opt=torch.optim.Adam(policy.parameters(),lr=cfg.policy_trainer.lr,betas=cfg.policy_trainer.betas)
            opt.load_state_dict(copy.deepcopy(ck['policy_opt']))
            assign_gradient(parameters,g)
            torch.nn.utils.clip_grad_norm_(parameters,cfg.policy_trainer.grad_clip_norm,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_value_(parameters,cfg.policy_trainer.grad_clip_value)
            opt.step()
            with torch.no_grad():after=policy(obs,goal)
            report['actor_one_step_counterfactuals'][key]=dict(action_delta_rms=float((after-initial_action).square().mean().sqrt()),
                action_abs_mean=float(after.abs().mean()),learning_rate=opt.param_groups[0]['lr'])
            policy.zero_grad(set_to_none=True)
        # Critic-only update on a frozen imagined batch with fixed original targets.
        if args.iteration==critic_it:
            critic.requires_grad_(True)
            opt=torch.optim.Adam(critic.parameters(),lr=cfg.policy_trainer.critic_lr)
            opt.load_state_dict(copy.deepcopy(ck['critic_opt']))
            X=[x.detach().reshape(-1,x.shape[-1])[valid.reshape(-1)] for x in (feat[:-1],ro.goals[:-1],xi[:-1],ro.e_root[:-1])]
            Y=targets[valid].detach();nvalid=len(Y)
            with torch.no_grad():before=critic(*X);before_mse=float((before-Y).square().mean())
            for epoch in range(cfg.policy_trainer.critic_epochs):
                perm=torch.randperm(nvalid,device='cuda:0')
                for indices in torch.tensor_split(perm,cfg.policy_trainer.critic_minibatches):
                    loss=torch.nn.functional.mse_loss(critic(*(x[indices] for x in X)),Y[indices])
                    opt.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(),cfg.policy_trainer.critic_grad_clip,error_if_nonfinite=True)
                    opt.step()
            with torch.no_grad():after=critic(*X)
            report['critic_frozen_batch_update']=dict(valid_states=nvalid,steps=20,
                loss_before=before_mse,loss_after=float((after-Y).square().mean()),
                predictions_before=stats(before),predictions_after=stats(after),fixed_targets=stats(Y),
                actor_has_no_gradient=all(p.grad is None for p in policy.parameters()))
            assert report['critic_frozen_batch_update']['actor_has_no_gradient']
        save(out/f'{label}.json',report)
        print(json.dumps({k:report[k] for k in ('objective','values_at_actual_starts','targets','gradient_norms','value_to_reward_gradient_norm_ratio','reward_value_gradient_cosine')},indent=2),flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('reports/critic_gradient_audit_20261030'))
    p.add_argument('--iteration',type=int,required=True)
    p.add_argument('--critic-iteration',type=int)
    main(p.parse_args())
