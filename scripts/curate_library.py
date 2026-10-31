"""Build an inspectable filtering manifest from converted references; no data deletion.

Edit keep/reason after visual review. Keyword filter is a conservative baseline,
not the unpublished exact paper split. Family grouping must be supplied for split generation.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from supermimic.data.quality import validate_reference

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("directory",type=Path)
p.add_argument("output",type=Path)
p.add_argument("--excluded-words",default="stair,ladder,chair,climb,box")
p.add_argument("--max-limit-excess",type=float,default=.05)
p.add_argument("--max-penetration-fraction",type=float,default=.05)
a=p.parse_args()
rows=[]
for path in sorted(a.directory.glob("*.npz")):
    reasons=[]
    try:
        with np.load(path,allow_pickle=False) as d:
            validate_reference(d)
            m=json.loads(str(d["metadata_json"].item()))
            q=m["quality"]
            if q["max_joint_limit_excess_rad"]>a.max_limit_excess:
                reasons.append("joint_limit")
            if q["foot_penetration_frames_2cm"]/len(d["joint_pos"])>a.max_penetration_fraction:
                reasons.append("ground_penetration")
            for word in filter(None,a.excluded_words.split(",")):
                if word.casefold() in path.stem.casefold():
                    reasons.append("keyword:"+word)
    except (ValueError,KeyError) as exc:
        reasons.append("invalid:"+str(exc))
    rows.append({"name":path.stem,"keep":int(not reasons),"reason":";".join(reasons)})
if not rows:
    raise ValueError("No NPZ reference clips")
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=["name","keep","reason"])
    writer.writeheader();writer.writerows(rows)
print(f"{sum(r['keep'] for r in rows)}/{len(rows)} passed initial screening; review {a.output}")
