# CuboiLeo 全库、论文观测实现的原生23DoF适配

基线：[CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2)，commit `ba6105349069bf040f1fdad60873d13138d505a2`，核查日期2026-10-29。这是第三方复现，不是SuperMimic作者发布的代码。逐文件哈希与状态见 `evidence/cuboileo_module_audit.json`：41个上游Python模块均有本地对应，其中18个一致，23个包含原生23DoF或环境接口等适配。

正式入口为 `configs/bones_full_2gpu.json`。使用全部129,477条已筛选原生23DoF动作，不划分holdout，`anchor_obs=false`。保留论文规模每卡Actor4096；上游e8实验实际为4卡、每卡Actor2048，本项目不是其训练命令的逐字复制。此前版本备份位于 `reports/cuboileo_adaptation_backup/`。

## 论文章节 → 本地代码

| 论文/上游实现项 | 本地代码 | 适配结果 |
| --- | --- | --- |
| 附录E动作准备 | `scripts/prepare_bones_seed_batch.py`、`src/supermimic/data/` | 沿用已完成的SOMA→原生23→按时间插值50Hz→附录E筛选；额外23DoF检查仍只作诊断，没有重新重定向或剔除数据 |
| 全库紧凑存储 | `data/packed_library.py`、`envs/mdp/motion_library.py` | 新增有界并发打包，fp16的59维状态，安全mmap读取，GPU常驻，按需torch FK；manifest、模型哈希、关节顺序、时间轴校验，失败不发布成品 |
| K=10目标、回报驱动重置 | `envs/mdp/commands.py`、`motion_library.py` | 1秒分箱；50%均匀、40%困难softmax(-25G/Gmax)、10%记忆；每卡全库采样 |
| 表A.1–A.3观测、奖励、终止与ξ | `envs/mdp/{observations,rewards,terminations}.py`、`envs/{sim_context,state_io}.py` | 关节维度改为23；无额外anchor观测；保留8项奖励，pelvis高度/倾斜硬终止，摩擦/关节零偏/躯干CoM上下文 |
| 可微运动学 | `kinematics/g1_fk.py` | 本地23DoF模型，位置/朝向/速度均为torch，MuJoCo用于对齐核查 |
| §3.2世界模型、Eq.1 | `world_model/` | 航向规范化状态、MLP增量与显式积分、学习终止头、解析奖励/观测；连续8步自回归训练；开启torch.compile |
| 表A.4策略与价值网络 | `policy/nets.py` | 目标编码器+tanh32维潜变量+历史观测解码器；23维动作；Critic使用状态、目标、ξ及根误差 |
| §3.3 Eq.2–3 | `imagination/` | H=32反传，冻结WM/Critic参数但保留状态梯度，软生存权重、末端V、TD(λ=.95)；编译前后输出和梯度测试通过 |
| 算法A.1、表A.5–A.6 | `training/`、`scripts/train_bones_2gpu.sh` | UTD .4，余弦学习率，torchrun每卡仿真/Replay，梯度平均，LR×√world；续训重新执行完整warmup |
| 附录F完整动作评测 | `eval/benchmark.py`、`sm23 evaluate` | 新增packed全库入口，跟随checkpoint观测设置，从首帧播放，失败原因及完整时长回报；与训练使用同一manifest |
| §4.3扰动诊断 | `scripts/eval_push.py` | 新增原生23 T-pose、固定随机方向的速度扰动、零扰动基线、失败锁存；恢复阈值是社区诊断选择 |
| 漂移与交互回放 | `scripts/diag_drift.py`、`scripts/play.py`、`eval/viewer_adapter.py` | 移植上游工具，跟随23DoF checkpoint；限定显式选片；Viewer读取缓存观测，避免重复推进历史，无rsl_rl训练器 |

表中模块路径省略共同前缀 `src/supermimic/`。运动控制训练、可微FK和BPTT由torch实现；物理后端仍为MuJoCo Warp，离线格式处理使用NumPy/SciPy。并非将物理引擎改为torch。

## 网络、权重与训练规模

| 项目 | 当前配置 |
| --- | --- |
| 策略观测 | 每帧23关节位置+23速度+3根角速度+3重力+23上一动作=75维，10帧历史=750维 |
| 未来目标 | 每帧23关节位置+23速度+3重力+3根角速度+3根线速度+1高度=56维，10帧=560维 |
| 动力学 | 59维规范化状态+23动作+27维ξ→2048/2048/2048→58维增量 |
| 终止头 | 4→64→64→1，sigmoid终止概率，BCE权重1 |
| Encoder / Decoder | 各2048/1024/512/256，latent32+tanh，动作输出23 |
| Critic | 256/128→1 |
| 每卡环境 / Replay | 4096 / 每环境1000步 |
| 每次WM更新 | 10批×2048窗口，窗口8步，梯度范数上限1 |
| 每次Actor更新 | 4096条×32步，Adam β=(.9,.95)，范数裁剪1后元素裁剪.1 |
| Critic更新 | 5 epoch×4 minibatch，γ=.99，λ=.95 |
| 学习率与计划 | 基础2e-4，双卡自动×√2；总320k步，250k步余弦衰减至.03倍；UTD .4 |
| 身体位置/朝向/线速度/角速度奖励 | 权重各1；尺度分别.3/.4/1/3.14 |
| 根位置/根朝向奖励 | 权重各.5；尺度.3/.4 |
| 动作变化 / 关节限位惩罚 | -.1 / -2；总奖励乘dt=.02 |

未来目标字段、规范化/增量编码、终止头隐藏层、SiLU、初始化等是CuboiLeo对论文未完整公开部分的实现选择。不能将这些细节称为作者官方规定。23DoF输入/输出维度是机器人适配；隐藏层宽度没有随自由度缩小。

### 参数、归一化和初始化的进一步核对

`evidence/cuboileo_parameter_audit.json`逐项记录76个可直接读取的源码默认值，均与上述固定版本一致。该核对覆盖源码默认值，不等同于取得了e8运行时保存的完整config。两个正式JSON配置现已显式保存环境、网络、状态编码、优化器和调度的原有默认值；展开前后的解析结果完全一致，便于以后检查默认值漂移。

| 细节 | 采用的CuboiLeo设置 |
| --- | --- |
| 隐藏层激活 | SiLU；编码器输出单独使用tanh |
| 动力学输出层初始化 | 权重、偏置均为0；初始模型做恒速显式积分 |
| 策略解码器输出层初始化 | 正交初始化gain=.01，偏置0；不是载入29DoF训练权重 |
| WM误差尺度 | 根位置/旋转向量/线速度/角速度/关节位置/关节速度：`.05, .1, .5, 1, .1, 2` |
| WM输入尺度 | 根高度/6D旋转/航向系线速度/本体系角速度/关节位置/关节速度：`1, 1, 1, 2, 1, 5` |
| WM输出尺度 | 线速度增量/角速度增量/关节速度增量/位置修正/旋转修正/关节位置修正：`.3, .5, 2, .01, .02, .02` |
| 策略观测除数 | 关节位置/关节速度/根角速度/重力/上一动作：`1, 5, 2, 1, 1` |
| 目标观测除数 | 关节位置/关节速度/重力/根角速度/根线速度/根高度：`1, 5, 1, 2, 1, 1` |
| 优化器 | 全部Adam；Actor β=(.9,.95)，WM/Critic默认β=(.9,.999)，eps=1e-8，weight_decay=0 |
| 损失组合 | 每步对缩放后的58维状态误差平方求和、对batch平均，再对8步平均；加权重1的终止BCE |
| Actor目标尺度 | 负的软生存加权目标（包含末端V）除以H=32；未另加PPO熵项或KL项 |

WM误差除以尺度后再平方，所以它隐含了不同物理量的权重；例如根位置系数为`1/.05²=400`，不是所有物理量未经归一化直接等权相加。23DoF适配改变关节项数量，没有另外添加29/23补偿系数。

奖励权重及尺度仍由实际代码 `envs/mdp/rewards.py:TrackingRewardCfg` 和 `envs/tracking_env_cfg.py` 定义，审计JSON只是可读快照，不是第二个生效的奖励配置。仿真与可微奖励沿用已通过逐项对齐测试的数值。

## 保留的本地修复与区别

- 使用t+1…t+10的未来目标；执行动作后，以s(t+1)比较reference(t+1)。N帧只有N−1次转换。保留奖励前刷新FK、终止前记录状态，以及Replay终止标签对齐修复。
- 想象展开到片段末端后停止生存权重，Critic不拟合反复复制的末帧。根奖励锚点使用pelvis，上游默认使用torso。
- fp16根四元数取出后重新归一化。缓存根角速度是世界坐标，WM/MuJoCo状态是本体坐标，通过接口转换，不能直接混用同为59维的数组。
- 主评测分母为完整参考的`0.1*(N-1)`，提前终止的余下回报为0。上游使用实际执行步数；额外保留`executed_step_norm_return`和其均值用于对照，失败时这两个口径不等价。
- 全库缓存新增结构校验，不使用诊断标记额外剔除动作。实际129,477条与论文129,514条不同，不强凑数量。

## 数据与运行

已生成 `data/training/bones_seed_v4_full_v1/library_native23_fp16.pt`：129,477条、47,329,219帧、262.2208小时，状态张量5.2013GiB。所有参考NPZ均已逐数组校验；报告在同目录 `.json`。量化误差逐列记录，其中根位置最大误差3.90625mm。校验通过表示表示格式/时间轴一致，不代表每条动作已通过动态控制验证。

以后重建使用新输出路径，避免覆盖现有成品：

```bash
.venv/bin/sm23 pack data/retargeted/bones_seed_native23_v4/library_50hz \
  data/training/bones_seed_v4_full_v1/library_native23_fp16_new.pt \
  --manifest data/training/bones_seed_v4_full_v1/train.csv --workers 8
```

迁移训练仅需代码/机器人资源、manifest和`.pt`缓存；NPZ原件保留用于追溯、选片回放及重新打包。两卡各自加载完整参考库，显存不合并。

```bash
# 在两张96GB卡的目标机器上；入口含GPU绑定及缓存/manifest核对。
bash scripts/train_bones_2gpu.sh --run-name bones_2gpu_hardwarecheck --max-hours .25
bash scripts/train_bones_2gpu.sh

# 正式训练完成后的全库覆盖评测。
.venv/bin/sm23 evaluate logs/bones_full_2gpu/ckpt_latest.pt \
  --data data/training/bones_seed_v4_full_v1/library_native23_fp16.pt \
  --manifest data/training/bones_seed_v4_full_v1/train.csv \
  --num-envs 1024 --output reports/bones_full_benchmark.json

# 以下工具需要训练后的checkpoint；具体片段名称来自manifest。
.venv/bin/python scripts/eval_push.py --ckpt logs/bones_full_2gpu/ckpt_latest.pt
.venv/bin/python scripts/diag_drift.py --ckpt logs/bones_full_2gpu/ckpt_latest.pt \
  --clips 210531__jump_and_land_heavy_001__A001
.venv/bin/python scripts/play.py --ckpt logs/bones_full_2gpu/ckpt_latest.pt \
  --clips 210531__jump_and_land_heavy_001__A001 --viewer native
```

## 本机验证范围

- 原有及新增缓存/编译/恢复测试66项通过；新增packed评测队列与提前失败分母测试另1项通过。
- 全129,477条缓存实际加载到GPU，以完整网络宽度、k8、H32、编译开启，完成102次采样、1次WM/Actor–Critic更新并保存检查点。这个检查使用4环境、WM批量4×1、Actor批量2；正式配置没有缩小。日志与隔离配置见 `reports/cuboileo_full_library_smoke*`，不能将这次运行称为论文规模训练或已收敛。
- 扰动脚本完成2环境、两种速度、0.3秒流程检查；漂移脚本完成1条真实动作检查。检查点只训练过1次更新，报告不是控制性能结论。
- 交互Viewer入口已适配、CLI可加载；本轮没有打开OpenGL窗口进行人工视觉验收。
- 后续部署更新：双96GB机器已连接；完成NCCL运行库适配和全库、每卡4096环境/Actor4096的短检查，正式长训练已启动。长期稳定性和收敛仍需观察，详见[远端训练](remote_training.md)。
