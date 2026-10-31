"""Evaluate presentation candidates; this deliberate selection is not a benchmark."""
import argparse,json,re
from dataclasses import asdict
from pathlib import Path
import torch
from supermimic.data.packed_library import load_pack,validate_pack
from supermimic.eval.benchmark import load_policy_from_checkpoint,run_benchmark
from supermimic.kinematics.rotations import quat_apply_inverse,yaw_quat


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    source=load_pack(a.library);groups={}
    patterns={'walk':r'__walk_ff_loop_', 'frog':r'__frog_jump_', 'jog':r'__jog_ff_loop_',
              'dance':r'__dance_hiphop_prep_', 'kick':r'__.*kick.*', 'punch':r'__pels_air_punch_'}
    for group,pattern in patterns.items():
        choices=[]
        for i,name in enumerate(source['names']):
            n=int(source['lengths'][i])
            if name.endswith('_M') or not re.search(pattern,name) or not 140<=n<=650:continue
            r=source['ref'][int(source['starts'][i]):int(source['starts'][i])+n].float()
            # A forward filename alone is insufficient: verify velocity in root heading coordinates.
            v=quat_apply_inverse(yaw_quat(r[:,3:7]),r[:,30:33]);forward=float(v[:,0].mean())
            if group in ('walk','jog') and (forward<.25 or float(v[:,1].abs().mean())>forward*.5):continue
            choices.append((i,name,forward))
        # Keep all frog candidates, diverse bounded others, stable dataset ordering.
        if group!='frog' and len(choices)>6:
            choices=[choices[i] for i in torch.linspace(0,len(choices)-1,6).long().tolist()]
        groups[group]=choices
    picked=[x for group in groups.values() for x in group];chunks=[];lengths=[];starts=[]
    for i,_,_ in picked:
        n=int(source['lengths'][i]);start=int(source['starts'][i]);starts.append(sum(lengths));lengths.append(n);chunks.append(source['ref'][start:start+n])
    pack={**source,'names':[x[1] for x in picked],'starts':starts,'lengths':lengths,'ref':torch.cat(chunks),'manifest_sha256':None}
    validate_pack(pack);torch.save(pack,a.output/'candidates.pt');del source,pack,chunks
    policy,_=load_policy_from_checkpoint(a.checkpoint,'cuda:0')
    rows=run_benchmark(policy,[str(a.output/'candidates.pt')],num_envs=len(picked));by={r.name:r for r in rows}
    selected=[]
    for group,choices in groups.items():
        good=sorted([by[n] for _,n,_ in choices if not by[n].failed],key=lambda r:-r.norm_return)
        print(group,[(r.name,round(r.norm_return,3)) for r in good],flush=True)
        for j,r in enumerate(good[:2]):selected.append({'slug':group if j==0 else f'{group}2','category':group,'name':r.name,'screening':asdict(r)})
    report={'purpose':'Hand-selected demonstration; not an unbiased evaluation','candidates':[asdict(r) for r in rows],'selected':selected}
    (a.output/'selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
