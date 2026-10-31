# 浏览器 Demo

[在线运行](https://yezheng04.github.io/supermimic-g1-23dof/demo/)：选择行走、蛙跳、慢跑、街舞、踢腿或出拳，再选择 80k / 104k 策略。每类两条精选动作，共12条。支持暂停、重置、轻推以及正面/侧前方/侧面视角；拖动画面可自由旋转。

这是浏览器内的真实闭环仿真。ONNX Runtime Web 每20ms仿真时间读取当前状态与历史，输出23维动作；MuJoCo WASM执行4个5ms物理子步。没有把预存的动作或状态轨迹当作策略。动作结束或触发原有根部高度/倾斜终止阈值后暂停，不自动接续下一条动作。硬件较慢时墙钟速度可能低于实时。

为了减少下载和计算，固定参考的未来10帧先通过对应checkpoint的原始目标编码器，保存32维编码。浏览器运行原始动作解码器，使用750维、10步、按观测项排列的实时历史。切换策略会同时切换解码器与目标编码，不能交叉使用。发布包不包含原始BVH、完整训练动作库或预存策略动作。

## 与训练的对应

- 保留原生23DoF顺序、默认姿态、动作尺度/偏移、PD增益、力矩限制、惯量、关节限位及全部碰撞几何。
- 物理XML从训练环境的`MjSpec`导出。视觉网格单独绘制，质量与惯量保持显式原值；导出时核对结构数组。不能使用仅含外观的裸机器人XML替代训练场景。
- 初始化使用参考首帧状态；观测无噪声、摩擦1.0，无域随机化。
- MuJoCo WASM **3.8.0**使用单线程构建，可直接部署GitHub Pages。训练为MuJoCo/Warp **3.5.0**；浏览器结果不能代替原后端评测。
- ONNX Runtime Web **1.23.2**；Three.js **0.180.0**。推理和物理在Web Worker中运行，页面隐藏或离开嵌入区后暂停。

本地Chromium验证：两个策略初始输出与PyTorch最大绝对误差均为`8.35e-7`以内；行走前60步与本机CPU MuJoCo的qpos最大绝对差`3.73e-7`以内。12条动作×2个策略的完整浏览器执行均未触发终止。精选是为展示而筛选，不能外推全库成功率；终止未触发也不等于姿态跟踪完美。

## 本地使用与重建

```bash
python3 scripts/download_demo_assets.py
python3 scripts/preview_website.py
# http://localhost:8000/demo/
```

下载脚本检查SHA-256后解包浏览器运行资源。只有点击首页启动按钮才加载交互引擎；策略模型按需加载并在当前会话复用。

拥有训练缓存和模型时，可重建：

```bash
PYTHONPATH=src .venv/bin/python scripts/select_showcase_motions.py \
  --library data/training/bones_seed_v4_full_v1/library_native23_fp16.pt \
  --checkpoint models/models_0080000.pt --output reports/showcase_new
PYTHONPATH=src .venv/bin/python scripts/export_browser_demo.py \
  --library reports/showcase_new/candidates.pt \
  --selection reports/showcase_new/selection.json \
  --checkpoint models/models_0080000.pt models/ckpt_0104000.pt \
  --output reports/showcase_new/browser
npm ci --prefix website/demo
python3 scripts/package_browser_runtime.py --assets reports/showcase_new/browser
```

104k的浏览器解码器已在运行资源包中；以上重建命令需要自行准备含完整Actor的104k checkpoint。导出同时生成PyTorch输入/输出样例与CPU逐步校验轨迹。安装Playwright及Chromium后，可运行 `node scripts/check_browser_demo.cjs`，通过 `DEMO_URL` 指定页面地址；该脚本验证嵌入、策略/动作切换、暂停、重置、扰动反馈、视角和手机布局。

## 来源

参考[SuperMimic的Web demo入口](https://supermimic.github.io/#web-demo)，浏览器代码独立实现，未复用官方受限制的脚本或权重。[MuJoCo官方WASM](https://github.com/google-deepmind/mujoco/tree/3.8.0/wasm)、[ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)、Three.js的许可位于`licenses/browser/`。

Training data includes [Motion Data by Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.
