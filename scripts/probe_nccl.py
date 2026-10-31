"""Two-rank collective correctness probe without the simulator or optimizer."""
from supermimic.scripts import train  # GPU visibility before importing torch/warp
import datetime
import json
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
torch.cuda.set_device(0)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=45))
try:
    print(json.dumps(dict(rank=rank, visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=torch.cuda.get_device_name(0), nccl=torch.cuda.nccl.version())), flush=True)
    for n in (1, 59, 8192, 8_000_000):
        x = torch.full((n,), rank + 1., device="cuda")
        print(f"rank={rank} before all_reduce n={n}", flush=True)
        dist.all_reduce(x, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        assert (x == 1.5).all()
        x.fill_(rank)
        dist.broadcast(x, src=0)
        torch.cuda.synchronize()
        assert (x == 0).all()
        print(f"rank={rank} PASSED n={n}", flush=True)
finally:
    dist.destroy_process_group()
