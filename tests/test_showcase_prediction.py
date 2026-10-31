"""The display intervention must affect imagined controls without future-state input."""
import importlib.util
from pathlib import Path
import sys

import torch

scripts=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('showcase',scripts/'record_showcase.py')
showcase=importlib.util.module_from_spec(spec);spec.loader.exec_module(showcase)


def test_held_action_and_current_friction_are_used_for_entire_prediction():
    contexts=[]
    class Dynamics:
        def step(self,state,action,xi):
            contexts.append(xi.clone())
            out=state.clone();out[:,7:30]+=action
            return out
    class Library:
        def last_frame_of(self,frames):return torch.full_like(frames,100)
        def goal_window(self,frames,n):return torch.zeros(len(frames),560)
    state=torch.zeros(2,59);state[:,3]=1
    xi=torch.zeros(2,27);xi[:,0]=torch.tensor([1.,.3])
    history=torch.zeros(10,2,75)
    held=torch.full((2,23),.25)
    poses,first=showcase.future_rollout(Dynamics(),lambda h,g:torch.ones(2,23),Library(),state,
        history,xi,torch.zeros(2,dtype=torch.long),torch.zeros(23),torch.tensor([False,True]),held)
    assert first[:,0].tolist()==[1.,.25]
    assert poses[0,:,7].tolist()==[8.,16.,24.,32.]
    assert poses[1,:,7].tolist()==[2.,4.,6.,8.]
    assert len(contexts)==32 and all(torch.equal(c,xi) for c in contexts)
