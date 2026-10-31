import torch
from supermimic.data.replay_buffer import ReplayBuffer


def test_done_belongs_to_transition_not_following_action():
    b=ReplayBuffer(1,16,1,1,1,"cpu")
    ids=torch.tensor([0])
    for t in range(6):
        v=torch.tensor([[float(t)]])
        b.add(ids,v,v,v,v,torch.tensor([0.]),torch.tensor([float(t>=4)]),
              torch.tensor([t]),torch.tensor([0]),torch.tensor([t==5]),torch.tensor([t==0]))
    w=b.sample_windows(256,3)
    torch.testing.assert_close(w.done,(w.frame==4).float())
    assert (w.states[1:]-w.states[:-1]==1).all()
    b.end_episode(ids)
    for t in range(4):
        v=torch.tensor([[100.+t]])
        b.add(ids,v,v,v,v,torch.tensor([0.]),torch.tensor([0.]),torch.tensor([t]),
              torch.tensor([1]),torch.tensor([False]),torch.tensor([t==0]))
    w=b.sample_windows(256,3)
    assert (w.states[1:]-w.states[:-1]==1).all()  # No 5->100 transition.


def test_ring_wrap_cannot_join_newest_to_oldest():
    b=ReplayBuffer(1,8,1,1,1,"cpu")
    ids=torch.tensor([0])
    for t in range(18):
        v=torch.tensor([[float(t)]])
        b.add(ids,v,v,v,v,torch.tensor([0.]),torch.tensor([0.]),torch.tensor([t]),
              torch.tensor([0]),torch.tensor([False]),torch.tensor([t==0]))
    w=b.sample_windows(64,3)
    assert (w.states[1:]-w.states[:-1]==1).all()
