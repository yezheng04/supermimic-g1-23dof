# BONES-SEED 真实动作训练

重定向结果位于 `data/retargeted/bones_seed_native23_v4`。训练只使用原清单中 `keep=1` 的 129,477 条动作；额外23DoF诊断不再筛掉动作。原始重定向输出不修改。

## 正式协议：全库训练与训练集覆盖评测

按[论文附录E、F.1–F.3](https://arxiv.org/pdf/2610.28480)，筛选后的动作全部进入训练，并在同一动作库上从首帧完整评测。原文129,514条是训练集基准，不是独立测试集；主要指标衡量训练动作覆盖程度。

我们的正式训练与评测均使用129,477条保留动作，不再划出独立验证/测试集。入口为 `configs/bones_full_v1.json`，清单为 `data/training/bones_seed_v4_full_v1/train.csv`；`protocol.json`明确训练与基准使用同一清单。正式配置现已恢复论文单GPU规模：4096环境、每环境1000步Replay、每次WM更新10批×2048窗口、Actor每次4096条32步展开。配置已核查加载清单，2026-10-29已在双RTX PRO 6000上启动全量训练。历史pilot和 `rtx5080.json` 的小批量设置不代表正式入口。

用 `scripts/probe_paper_scale.py` 可以进行一次隔离的规模测试：用12条真实动作生成Replay，但保持论文环境数、网络、WM批量和Actor批量。若小动作库已经显存不足，全库也无法以当前实现直接运行；若通过，可加 `--reserve-full-library --output reports/paper_scale_probe_full_reserve.json` 检查全库参考状态及索引的额外内存下界。这是显存测试，不是把正式训练库缩成12条，也不是全库训练成功的证明。

2026-10-29实测结果见 `reports/paper_scale_probe_v1.json`：RTX5080上，4096环境初始化、100步仿真采样、10批×2048窗口的WM更新通过；4096轨迹×32步的Actor–Critic更新发生CUDA OOM，错误报告进程占用约15.31GiB，设备空闲43.94MiB。测试仅载入12条动作，未预留全库参考内存。进程已退出并释放显存，正式参数保持不变。该结果只证明当前未采用显存优化的实现直接运行不够；没有测出所需最小显存，也不排除通过梯度累积或激活重计算保留有效训练规模。

终止概率是论文§3.1–3.3的软生存机制组成部分，BCE监督来自仿真硬终止标签；不是额外加入的新训练目标。当前终止头的4维误差特征和64/64隐藏层是工程具体化，论文未公开这组结构。仿真本身仍按高度/倾斜阈值硬终止。

## 双RTX PRO 6000 Blackwell（每卡96GB）

`configs/bones_full_2gpu.json`与`scripts/train_bones_2gpu.sh`准备了同机双卡入口。设计直接核对CuboiLeo上游commit `ba6105349069bf040f1fdad60873d13138d505a2`；两者的分布式辅助模块相同。每卡独立仿真、Replay、参考库和模型，每次反传后显式平均梯度。先按local rank限制可见GPU，再初始化CUDA，避免MuJoCo Warp的两个进程都使用同一物理卡。

| 项目 | 每卡 | 两卡合计 |
| --- | ---: | ---: |
| 环境数 | 4096 | 8192 |
| Replay名义容量 | 4,096,000行 | 8,192,000行 |
| WM窗口批量 | 每次10批×2048 | 每次10批×4096 |
| Actor想象轨迹 | 4096条×32步 | 8192条×32步 |

配置中的基础学习率仍为2e-4，Trainer在world_size=2时自动乘sqrt(2)，实际为约2.828427e-4；不要手工再乘一次。训练320k步、250k步余弦衰减；WM和ImagStep开启torch.compile，anchor_obs保持false。所有129,477条动作在每张卡上均可采样，不建立互斥验证集。

上游e8全库实验实际用了4卡、每卡Actor批量2048，不能当作论文每卡4096的原样设置。本配置保留论文批量。96GB/卡预计有较大余量，但未经目标机器实测；数据并行不将两卡显存合成一个192GB空间，也不保证复现论文训练时长。

目标机器需先准备本项目兼容的运行环境、原生23DoF资源，以及`data/training/bones_seed_v4_full_v1/library_native23_fp16.pt`和同目录`train.csv`。缓存已完成并经过全库逐数组校验，详见[CuboiLeo适配](cuboileo_adaptation.md)。训练不再要求目标机器同时保留全部NPZ；NPZ用于追溯与选片回放。不要直接复制旧机器虚拟环境或让它指向旧绝对路径。先查看两卡与拓扑，再做限时硬件检查：

```bash
nvidia-smi
nvidia-smi topo -m
# 在项目根目录执行；固定长学习率计划，另命名诊断运行。
bash scripts/train_bones_2gpu.sh --run-name bones_2gpu_hardwarecheck --max-hours 0.25

# 硬件检查通过后，从头正式训练（不自动载入pilot权重）。
bash scripts/train_bones_2gpu.sh

# 后续恢复，使用原始配置基础学习率。
bash scripts/train_bones_2gpu.sh --resume logs/bones_full_2gpu/ckpt_latest.pt
```

限时检查在log_interval处判断，首轮编译可能使其超过15分钟。启动器检查两张可见CUDA卡、数据路径和日志覆盖风险；可用`CUDA_VISIBLE_DEVICES=2,3`选择其他卡对，`SM23_PYTHON`指定目标机器解释器。正式主入口使用`supermimic.scripts.train --config`，保留GPU绑定逻辑。

本机只有一张5080，已验证全库缓存GPU加载、完整网络宽度的编译版单次WM/BPTT更新及训练/恢复回归；此次检查缩小了环境数和批量。远端现已通过27项运行测试及全库、论文每卡批量的双卡短检查，并启动正式训练；长期稳定性与收敛仍需观察。部署和NCCL运行库适配见[远端训练](remote_training.md)。

下面的90/5/5划分是此前额外加入的工程诊断，**偏离了原文主实验协议，现已撤回作为正式训练划分**。文件和结果仅保留用于解释已完成的小集实验，其中所有动作仍属于正式全量训练库。

## 历史诊断划分（不用于正式训练入口）

`data/training/bones_seed_v4_split_v1/` 保存全库索引和三个划分清单：

| 划分 | 动作数 | 命名家族数 | 50Hz帧数 |
| --- | ---: | ---: | ---: |
| train | 116,405 | 3,374 | 41,745,844 |
| validation | 5,589 | 182 | 1,969,977 |
| test | 7,483 | 193 | 3,613,398 |

随机种子为20261029。移除采集日期、演员后缀、镜像后缀和末尾三位take编号后，按家族名称SHA256分配90%/5%/5%；因此实际动作比例不会恰好等于这三个数字。同组包含跨日期、演员和镜像的变体。命名无法识别的语义别名仍可能需要人工归组，不能据此宣称已排除所有来源重复。

全库47,329,219帧，约262.22小时。FP16的59维状态约5.20GiB，FP32约10.40GiB，逐帧索引另占0.35GiB，均未含网络、优化器、Replay和仿真。先实测显存余量，再决定全库驻留或分块加载。不要对完整NPZ运行现有的 `sm23 pack` 后直接假定16GB显卡能承受所有训练配置。

早期仅检查了sidecar及18条pilot数组；2026-10-29构建fp16缓存时，已对全部129,477条保留NPZ完成数组、关节顺序、四元数、时间戳、模型哈希和有限值检查，未新增剔除动作。

只有另行研究未见动作泛化时才需要重新生成这种诊断划分；复现主实验不运行此步骤。脚本拒绝覆盖已有划分：

```bash
.venv/bin/python scripts/prepare_training_splits.py --output data/training/bones_seed_v4_split_v2
```

## 小规模运行

`pilot_train`、`pilot_validation`、`pilot_test` 是指向原参考库的符号链接，不复制或重新重定向动作。分别有12/6/6条动作，从走路、转身、挥手类别按名称和时长固定选取，每条属于不同命名家族。这是集成测试子集，不代表全库的类别分布；其中包含转身慢跑、蹲姿转身等组合动作。测试子集保留，当前不用于调参。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/run_bones_pilot.py train
.venv/bin/python scripts/run_bones_pilot.py resume
.venv/bin/python scripts/run_bones_pilot.py evaluate
```

`configs/bones_pilot_v1.json` 是200步短检查：32环境、WM批量16/每次2个minibatch、Actor批量8。主网络宽度、8步WM训练、32步BPTT、10帧未来目标保持原实现。检查点和报告分别在 `logs/bones_pilot_v1/`、`reports/bones_pilot_v1/`。脚本拒绝覆盖已有训练日志；重跑先复制配置并改 `run_name`。

延长检查使用 `configs/bones_pilot_extended_v1.json`：从短检查载入模型参数，重新建立优化器与Replay，WM批量32、每次10个minibatch；学习率按320,000步计划衰减，在每50步的检查点达到6分钟时间上限后结束。因此总耗时可略超6分钟，初始化和评测时间另计。

```bash
.venv/bin/python scripts/run_bones_pilot.py train --config configs/bones_pilot_extended_v1.json
.venv/bin/python scripts/run_bones_pilot.py evaluate --config configs/bones_pilot_extended_v1.json
```

`resume` 模式是单独的恢复诊断，会先按配置完成完整warmup再验证更新，结果写入另一个日志目录；性能评测使用原训练检查点。正式续训用原始长计划配置配合 `sm23 train --resume ...`，调整 `max_hours` 控制时长。

## 对齐检查和结果

前10万次的20k/40k/60k/80k/100k检查点比较，见[固定子集评测协议](checkpoint_evaluation_100k.md)。主样本与困难动作诊断分别统计，不以子集成绩代替全库成绩。

俯卧撑、爬行、尺蠖等手部支撑动作失败时，按[重定向审计中的排查约定](retargeting_audit.md#手部支撑动作的失败排查约定2026-10-29)对照人体源动作、23DoF参考和策略仿真执行，记录证据后再归因；额外23DoF检查继续仅作诊断。

```bash
.venv/bin/python scripts/audit_training_alignment.py \
  --data data/training/bones_seed_v4_split_v1/pilot_train \
  --checkpoint logs/bones_pilot_v1/ckpt_initial.pt \
  --output reports/bones_pilot_v1/real_motion_alignment.json
```

这项检查在真实动作上逐项比较仿真奖励与可微FK奖励，并核查参考帧推进、终止前状态和终止标签。完整片段评测从首帧开始，关闭随机扰动，提前失败的剩余时长按零回报计；不以随机起点短回合回报代替完整片段成功率。

每次训练保存独立验证环境采集的固定8步窗口，不用于训练更新。`training.json` 比较相同窗口上的初始/最终世界模型误差；`evaluation_comparison.json` 比较初始/最终策略在训练和验证动作上的仿真表现。世界模型误差下降与策略收敛是两个不同的验收条件。

详细历史运行结果见 `reports/BONES_TRAINING_START.md`。此前诊断测试子集尚未用于小集实验，但在正式协议下同样属于训练库。小集调试是工程步骤，不是论文规定的训练课程或独立测试协议。
