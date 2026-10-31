# SOMA原生23自由度试跑

日期：2026-10-27。当前已按用户选择移除GMR本地仓库及集成，使用SOMA Retargeter；以下GMR和混合路径结果仅为历史对照。

结论：人体BVH→Newton IK→原生23关节→50Hz/MuJoCo FK路径已实际运行；当前参数输出未通过几何质量验收，不是合格训练库。

已追加6段官方示例：挥手、街舞、跳跃、站立拾取、投球、伸展。完整双视角回放与逐动作质量统计见 [新增动作报告](../reports/SOMA_EXAMPLES.md)，总览视频约26秒。

移除GMR后的独立复验：26项回归检查通过（`reports/tests_soma_only.log`）；SOMA单独处理241帧人体输入，生成100帧50Hz参考（源采样跨度1.99992秒，输出1.98秒，余下不足20ms按规则记录）。结果在 `data/retargeted/soma_after_gmr_removal`，运行日志 `reports/soma_after_gmr_removal.log`。历史GMR实验结果和原始数据保留。

## 本地环境与模型

- 仓库：`<soma-root>`，分支 `feature/g1-23dof-retargeting`，提交 `6c0517236d98910a63a57b082c4a585b2776de91`。
- 独立环境：`<soma-root>/.venv`，Python3.12、Newton1.0.0、Warp1.12.0；用MuJoCo3.5.0做FK/QC。
- 目标直接使用本工程 `src/supermimic/robots/g1_23dof.xml`，没有用29关节轨迹删列。
- 23关节的名称及顺序已校验。4个抽样姿态的Newton/MuJoCo身体位置最大差异约1.36微米，见 `reports/soma_fk_parity.json`。
- 已下载公开随附的 `Neutral_walk_forward_002__A057.bvh` 以及 `soma_zero_frame0.bvh`；2026-10-28完整BONES-SEED uniform归档也已下载、校验并提取候选动作。
- 克隆时本机没有git-lfs；源码已正常checkout，所需两份LFS资产通过对应提交的GitHub media URL取得，其余大资产仍是指针。直接运行GUI可能还需要其他LFS资产。

## 已完成的适配

1. 自定义MJCF同时传给主IK和feet stabilizer。
2. 继承23分支的目标身体与权重；SOMA人体scaler双手局部x偏移补偿−0.084m，适配原生23腕端坐标原点。实际配置随结果保存为 `scaler_native23.json`。
3. 恢复BVH头的真实帧率：示例的 `Frame Time=0.008333` 对应120.004800192Hz，不能使用读取器取整后的120.0替代原始时基。
4. IK初始化/稳定前缀由上游裁掉，显式断言输出帧数等于所选输入帧数，防止前后错位。
5. 输出源频率NPZ、30列数字CSV（m/xyzw/rad）、SOMA带表头CSV（cm/欧拉度/关节度），各自注明约定。
6. 以物理秒重采样到50Hz，重算23模型FK、速度与质量报告。CSV帧率记录在旁边的 `motion.json`，不能从CSV列数推断。

## 运行命令

在 `<project-root>` 下执行，输出目录可自定：

```bash
../SOMA-Retargeter/.venv/bin/python scripts/retarget_soma.py \
  ../SOMA-Retargeter/assets/motions/bvh/Neutral_walk_forward_002__A057.bvh \
  data/retargeted/soma_native23_walk_new
```

已有结果位于 `data/retargeted/soma_native23_walk_calibrated/`：

- `raw.npz`：1088帧源频率23关节轨迹。
- `motion.csv`、`motion_soma.csv`：两种CSV表示，不能互换单位/旋转解释。
- `reference_50hz.npz`、`.json`：453帧、9.04秒；原输入采样跨度9.057971秒，尾部不足20ms已记录，没有拉伸。
- `preview.mp4`：运动学回放，不代表控制器成功跟踪。

## 真实检查结果

| 输出 | 50Hz帧数 | 硬限位违规帧 | 自碰撞穿透超过2mm的帧 | 足底低于地面2cm的帧 |
| --- | ---: | ---: | ---: | ---: |
| SOMA公开走路示例，完整1088源帧，腕端补偿后 | 453 | 0 | 169 | 48 |
| LAFAN走路前301源帧，实验性目标适配→SOMA IK | 500 | 0 | 500 | 0 |
| 同一LAFAN片段，GMR IK加增量碰撞约束 | 500 | 0 | 12 | 0 |

前两行与第三行不是统一人体预处理的算法排行榜；第一行动作本身不同。第二行虽复用了GMR人体目标标定，但IK权重、初始化与稳定器不同，出现严重右臂局部解失败。它只证明接口接通，**不能当作成功的LAFAN→SOMA人体骨架转换或合格动作**。

SOMA公开示例最大手髋穿透约7.51cm，足底最低约−3.31cm；最大关节速度约10.88rad/s。无硬限位违规不足以证明可跟踪，且脚部稳定器不负责全身自碰撞。当前全部输出标记 `training_approved=false`。

该实验性LAFAN混合路径现已移除。历史输出和上表数据保留用于对照，单动作重定向入口为 `scripts/retarget_soma.py`，全量批处理入口见下文，输入均为SOMA骨架BVH。

## GMR全量首轮结果

此前使用的GMR批处理脚本（现已移除）跑完原始77段BVH：827,691帧50Hz，累计16,552.28秒（约4.60小时）。每段都保存源频率NPZ/CSV及50Hz/FK/QC，目录 `data/retargeted/lafan1_native23_v1`。

77段都触发了至少一项整段几何筛查，`geometric_candidates.txt` 为空。该规则是整段零违规，因此不表示每个子片段都不可用；后续需修复或切分后重验，不能忽略报告直接训练。原始数据取自Ubisoft官方LAFAN1归档，来源和哈希在 `data/raw/lafan1_original/source.json`。

GMR碰撞约束使用优化增量形式，避免Mink1.3.0距离边界与delta_q变量之间的时间步缩放问题；单IK步关节增量限制0.1rad。它不是物理关节速度限制，也不保证帧间插值全程无碰撞，最终仍以50Hz QC为准。

本次相关回归验证：`reports/tests_retarget.log`，28项通过，包含30→50Hz物理速度/时长、BVH帧间隔精度、IK增量碰撞约束、FK与训练相位测试。

## 数据选择

[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) 由Bones Studio发布，提供142,220段（71,132原始+71,088镜像）、约288小时、标称120Hz动作，以及文本/分段/演员元数据。

| 数据表示 | 内容 | 本工程用途 |
| --- | --- | --- |
| `soma_uniform` | 所有动作使用统一人体骨架和比例的BVH | 优先作为SOMA重定向输入 |
| `soma_proportional` | 保留各演员体型比例的BVH及形状参数 | 需要重新处理比例标定 |
| `g1` | 已重定向的G1机器人CSV | 应先核对关节、单位、帧率；不是可默认直接使用的原生23参考库 |

完整SEED需在Hugging Face登录并接受该数据集许可。2026-10-28已下载并校验 `soma_uniform.tar.gz` 和元数据，归档身份见 `reports/bones_seed_archive_identity.json`。原始LAFAN1不能仅改骨骼名就认为符合SOMA局部坐标约定。

## BONES-SEED批量准备

参考 [SONIC数据准备说明](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html#prepare-robot-motion-data)，固定其仓库版本为 `1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b`，源码和许可证在 `references/gr00t_data/`。数据版本为 `2f59b2077b9da34dd4e43618e705c7cb962c9a66`。

SONIC筛选脚本是40个文件名关键词的大小写不敏感子串匹配。现已补充 [SuperMimic附录E](https://arxiv.org/pdf/2610.28480#page=15) 的名称示例和几何规则。附录只给出了名称示例，并未公开完整词表；新增 `box`、`jump_off_50cm`、`look_over_fence`，按规范化后的词边界匹配，避免把 `boxing` 或 `position` 当作箱子或坐下。原有SONIC匹配保持不变，不按论文数量调整词条。

当前142,220条元数据经过SONIC规则剩129,785条；上述新增名称规则再剔除138条，剩 **129,647条候选动作**。论文表A.7为名称筛选后129,667条、几何筛选后129,514条。我们的名称结果相差20条，未宣称还原作者完整名单；这也不是23自由度几何检查后的数量。

```bash
# 仅生成全量筛选表、候选表和分类计数，不需要加载GPU。
python3 scripts/prepare_bones_seed.py

# 下载后处理；--wait-download 可在下载完成前启动。
../SOMA-Retargeter/.venv/bin/python scripts/prepare_bones_seed.py --run --wait-download
```

默认输出 `data/retargeted/bones_seed_native23_v3/`；旧v1名单和v2严格门槛试跑保留用于比较。v3采用用户选定的策略：附录E决定保留，额外23自由度检查只作诊断。可复用已校验的提取结果与原始SOMA输出：

```bash
../SOMA-Retargeter/.venv/bin/python scripts/prepare_bones_seed.py --run \
  --reuse-retargeted-root data/retargeted/bones_seed_native23_v2
```

复用会检查机器人XML和BVH源文件哈希，拒绝已知的标定对照或避碰修正输出；重新计算筛选结果，不复用旧 `screening.json`。

- `selection.csv` / `candidates.csv`：每条动作的来源、镜像标记、take分组、SONIC词条与新增名称规则的独立命中记录；不会删除原始压缩包。
- `status.json`：当前阶段及错误；同一目录只允许一个任务运行。
- `pilot_selection.json` / `pilot_report.json`：8类动作各抽8条，以固定哈希顺序选取，共64条，整段重定向后检查。
- `clips/`：重定向原始输出、50Hz参考、FK/速度及 `appendix_e.json`、`screening.json`；失败日志在 `logs/`。附录E结果与额外原生23结果分别记录为 `appendix_e_keep`、`native23_quality_keep`。当前 `native23_quality_mode=diagnostic` 时，最终 `keep` 由附录E决定，额外问题写入 `diagnostics`；只有可选strict模式才取两者交集。
- `library_50hz/`：按当前模式保留的平铺NPZ/JSON，命名为 `session__motion` 避免跨录制日期重名。当前模式允许存在自碰撞或足部穿地诊断，不能称为严格质量全通过。全量处理时生成训练加载器兼容的 `manifest.csv`。

任务会先验证下载包大小和SHA256，再安全提取候选BVH。重定向读取真实 `Frame Time`，保存源时间轴，再插值到均匀50Hz；最后不足20ms的尾段单独记录，不拉伸动作，不使用官方示例的整数跳帧去做120→50Hz转换。

附录E几何筛选以世界地面z=0为准，最低身体高度是所有非world身体**link原点**的最小z。任一规则触发就记录剔除原因：

| 规则 | 论文明确的阈值 |
| --- | --- |
| 高处支撑 | 最低身体link高度连续至少0.6秒大于0.20m |
| 身体穿地 | 任一帧最低link高度小于−0.12m，或超过80%的帧小于−0.08m |
| 双脚悬空 | 直立、站立高度且近乎静止时，两脚碰撞几何距地面均大于0.02m，连续至少0.5秒 |

双脚距离通过MuJoCo `mj_geomDistance` 测量，每只脚取其7个碰撞几何到z=0平面的最小有符号距离；不是脚踝高度，也不是两脚距离的平均。只测量符合姿态/静止门控的帧。筛选代码在 `src/supermimic/data/appendix_e.py`，使用原生23模型，不修改训练模型。

论文没有公开完整姿态/静止门控数值。当前在配置 `appendix_e_geometry.floating_gate_local` 明确采用以下实现选择：骨盆朝上余弦>0.9、高度在(0.60,0.85)m、线速度模长<0.1m/s、角速度模长<0.5rad/s、关节速度RMS<0.5rad/s。除角速度门控为本地增加，其余默认值参考已有 [Cuboi复现](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/scripts/data/filter_bones_seed.py)，不能标成原作者参数。持续时长按连续控制样本数/fps计算（50Hz下30帧和25帧）；作者未公开边界帧计数实现。报告携带全部配置并标记 `exact_author_implementation=false`。

额外原生23诊断阈值保持在 `quality_gate`：硬限位超出不超过1e-5rad、自碰撞穿透超过2mm的帧数为0、足部穿地超过2cm的帧数为0。这些是本工程的诊断规则，**不是附录E筛选项**；当前不因此剔除整段。有限值、关节顺序、单位四元数及时间网格仍必须验证通过，损坏或不兼容文件不能进入训练库。全量报告分别统计附录E通过数、额外诊断无异常数和最终保留数；速度与加速度仍记录供检查，未进行电机力矩、平衡或动态跟踪验收。

v2的64条试跑全部通过附录E，只有5条同时满足额外诊断阈值。此前以10%通过率停止批处理的门槛来自本地工程选择，并非论文要求。根据用户选择，v3已取消诊断模式下的这一停止条件，继续全量转换；`minimum_pilot_pass_fraction` 只在显式选择strict模式时生效。

通过运动学筛选的片段仍标记 `dynamic_feasibility=not_evaluated`、`training_approved=false`；最终动态可跟踪性需训练后在MuJoCo Warp固定条件下整段评测。修改筛选或重定向实现后，使用新的 `--work-dir`，避免混入旧结果。

## 常驻GPU批量准备

2026-10-28起，全量任务使用v4批处理目录。已完成的v3输出校验后复用，不重新计算IK。随后按用户要求清理v1–v3及试验目录；解压校验凭据已保存在v4内，后续续跑不再依赖v3。上面的旧版本命令仅作历史记录。

```bash
../SOMA-Retargeter/.venv/bin/python scripts/prepare_bones_seed_batch.py \
  --batch-size 64 --cpu-workers 8 \
  --reuse-retargeted-root data/retargeted/bones_seed_native23_v4
```

默认输出 `data/retargeted/bones_seed_native23_v4/`。同一命令再次运行会续接已完成结果；`--max-new-clips 128` 可做限定规模试跑。更改转换代码、模型、配置或运行库版本后，需要新的 `--work-dir`。已下载归档的校验记录和提取结果复用，启动时检查候选文件是否齐全。

清理后的便捷续跑入口为 `./scripts/resume_bones_seed_v4.sh`。现有任务运行期间不要重复启动；不要省略上述 `--reuse-retargeted-root`，旧批处理脚本的默认值仍指向已删除的v3，生产脚本因指纹校验保持不变。

- 一个常驻GPU进程维护Newton模型、IK求解器和CUDA graph；每个环境处理一条独立动作，每批重置求解状态。
- 按源帧数排序分批，默认最多64条，并限制 `最长源帧数×批量` 为131,072；超长动作缩小批量。目标、脚部稳定器中间结果和轨迹缓冲留在GPU，每批结束才拷回结果。
- 8个常驻CPU进程并行进行物理时间插值、50Hz FK/速度、诊断、附录E筛选和压缩存储。待导出队列有上限，不随全量数据增长。
- 每条新动作保存 `raw.npz`、`reference_50hz.npz/.json`、`appendix_e.json` 和 `screening.json`。批处理不额外生成两份冗余CSV；单动作入口仍支持CSV。
- `status.json` 记录完成数、CPU待处理数、吞吐量及粗略剩余时间；`batch_timings.jsonl` 记录每批加载/目标生成/求解耗时。后台任务位置记录在 `reports/bones_seed_preparation_job.json`。
- 对任务PID发送SIGTERM会在当前批次完成并排空CPU导出队列后退出；中断留下的未完成目录保留为 `.incomplete-*`，下次重算该条。不要同时在同一输出目录启动两个任务。

校准、IK迭代数、脚部稳定器、原生23模型和筛选策略沿用串行版本。部分uniform BVH的参考偏移不同，保留各文件原始骨架偏移和逐帧局部变换，仅复用拓扑一致的映射参数。不会把这些差异覆盖成首条动作的参考姿态。

64条完整动作对照耗时40.82秒（包含加载、GPU求解、CPU导出及筛选），原始IK、50Hz时间轴、关节、身体FK/速度的最大绝对差异均为0，筛选结论一致。结果见 `reports/soma_batch_benchmark.json`；这组动作长度混合，不能直接把短片段每秒条数当作全库速度。全量ETA根据实际处理源帧数持续更新。

GPU实现部分改编自SOMA NewtonPipeline与FeetStabilizer，保留上游版权声明及 [Apache-2.0许可证](../licenses/SOMA-Retargeter-Apache-2.0.txt)。训练仿真仍使用MuJoCo Warp；这里的Newton仅用于SOMA运动学重定向。

## 原生23模型与标定核查（2026-10-28）

核查可通过 `.venv/bin/python scripts/audit_native23_retargeting.py` 重现，结果在 `reports/native23_audit/summary.json`。

- **模型**：本工程与Unitree `g1_23dof.xml` 的SHA256完全一致。没有缩小碰撞几何或增加碰撞排除来提高通过率。走路最严重帧的可视化在 `reports/native23_audit/worst_walk.png`，手部确实进入髋部附近，不只是微小数值误差。
- **腕部偏移**：在50组共同关节姿态下，以共同肘部坐标系比较，29自由度wrist_yaw到23自由度wrist_roll原点差为局部x方向84mm，误差小于4e-16m。原先−84mm标定保留。
- **足部测量**：从64条动作各取5帧、两脚共14个碰撞几何，4480次 `mj_geomDistance` 与原胶囊支撑高度公式对比，最大误差小于3e-17m。足部穿地统计没有把脚踝原点误当足底。
- **标定对照**：4条代表动作分别测试取消腕偏移、手姿态权重降至25%、关闭脚部稳定器，共12次整段重定向。三种变化都没有稳定改善，部分动作更差；保持原默认标定。对照结果与日志在 `data/retargeted/native23_calibration_ablation/`。
- **阈值敏感性**：自碰撞容差从2mm放宽至10mm，仅使无该项异常的动作从6/64增加至12/64；放到5cm才到37/64。不能把普遍碰撞归因于阈值略严。足部2cm阈值下54/64无异常，但“一帧即淘汰整段”并非附录E规则，现仅诊断。
- **上游约束**：当前SOMA Newton IK路径没有创建自碰撞目标；脚部稳定器追踪脚部目标，不等于施加足底不得穿地的硬约束。未发现关节名称/顺序或Newton/MuJoCo坐标转换错误。

其他复现的核查依据：Cuboi和Kang的 `scripts/data/filter_bones_seed.py` 主要使用附录E名称与几何规则；MorphMimic的 `src/morphmimic/data/filters.py` 也如此，并有 `validate_csv.py` 和 `limit_stats.py` 提供完整性/限位统计。所查数据筛选脚本没有我们的2mm自碰撞、2cm足穿地任一帧即淘汰整段门槛。

独立避碰实验：`src/supermimic/retarget/collision_refine.py` 对手臂做限幅修正，保持根、腰、腿和时间轴不变。一个走路片段的自碰撞从262/296帧降至0帧，最大关节改变量约0.20rad；原有1帧足部穿地仍如实保留。输出在 `data/retargeted/native23_collision_refine_trial/walk/`。该实验**未接入默认重定向或全量数据**，尚未完成动态跟踪和多动作验证。
