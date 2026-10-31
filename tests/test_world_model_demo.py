import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("wm_demo", Path(__file__).parents[1] / "scripts/record_world_model_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_continuous_prediction_does_not_teacher_force_and_window_reseeds_at_boundary():
    class Dynamics:
        def step(self, state, action, context):
            return state + action
    states = (torch.arange(6)*10.).reshape(6,1,1)
    short, continuous = demo.predict(Dynamics(),states,torch.ones(5,1,1),torch.zeros(5,1,1),window=2)
    assert short.flatten().tolist() == [0.,1.,2.,21.,22.,41.]
    assert continuous.flatten().tolist() == [0.,1.,2.,3.,4.,5.]


def test_terminal_capture_retains_pre_reset_pose():
    class IO:
        state = torch.tensor([[1.],[2.]])
        def read(self):
            return self.state.clone()
    io=IO()
    capture=demo.TerminalCapture(io)
    io.state[0]=9
    capture.record_pre_reset(torch.tensor([0]))
    io.state[0]=0
    capture.record_post_reset(torch.tensor([0]))
    assert capture.states[0].item()==9


def test_future_trail_recomputes_policy_on_predicted_state():
    spec2=importlib.util.spec_from_file_location("future_demo",Path(__file__).parents[1]/"scripts/record_future_trail_demo.py")
    future=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(future)
    class Dynamics:
        def step(self,state,action,context):
            out=state.clone();out[:,7:30]+=action
            return out
    class Library:
        def last_frame_of(self,frame):return torch.full_like(frame,100)
        def goal_window(self,frame,k):return torch.zeros(len(frame),560)
    def policy(history,goal):
        return history[:,9*23:10*23]
    state=torch.zeros(1,59);state[:,3]=1;state[:,7]=1
    xi=torch.zeros(1,27);q0=torch.zeros(23)
    obs=future.analytic_observation(state,torch.zeros(1,23),xi,q0)
    states,actions=future.rollout_actor(Dynamics(),policy,Library(),state,obs[None].repeat(10,1,1),xi,torch.zeros(1,dtype=torch.long),q0,horizon=3)
    assert actions[:,0,0].tolist()==[1.,2.,4.]
    assert states[:,0,7].tolist()==[1.,2.,4.,8.]
