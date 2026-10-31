"""Stage pinned browser dependencies and exported assets; keep binaries out of Git."""
import argparse,shutil
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--node-modules',type=Path,default=Path('website/demo/node_modules'));p.add_argument('--assets',type=Path,required=True);a=p.parse_args()
out=Path('website/demo');vendor=out/'vendor';vendor.mkdir(parents=True,exist_ok=True)
files={'@mujoco/mujoco':['mujoco.js','mujoco.wasm'],
       'onnxruntime-web/dist':['ort.wasm.min.mjs','ort-wasm-simd-threaded.mjs','ort-wasm-simd-threaded.wasm'],
       'three/build':['three.module.js','three.core.js'],
       'three/examples/jsm/controls':['OrbitControls.js']}
for package,names in files.items():
    for name in names:shutil.copy2(a.node_modules/package/name,vendor/name)
shutil.copytree('licenses/browser',vendor/'licenses',dirs_exist_ok=True)
shutil.copy2(out/'NOTICE.md',vendor/'NOTICE.md')
shutil.copytree(a.assets,out/'assets',dirs_exist_ok=True)
print('Staged browser runtime and assets')
