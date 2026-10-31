# 公开项目的重定向与时间轴核查

当前方案：SOMA Retargeter，入口 `scripts/retarget_soma.py`，输入SOMA Uniform BVH。GMR本地仓库及项目集成已删除；下文GMR部分保留为历史核查记录，不代表当前可用入口。

## 手部支撑动作的失败排查约定（2026-10-29）

用户确认：后续俯卧撑、爬行、尺蠖等动作评测失败时，必须对照原始人体动作、23DoF重定向参考和策略在实际仿真模型中的执行效果，再判断是重定向、模型接触/自由度约束还是策略学习的问题。不能仅凭缺少腕部自由度认定重定向失败，也不能把所有失败直接归因于策略。

本次按动作名称核对 `data/retargeted/bones_seed_native23_v4/selection.csv`、`manifest.csv` 与最终 `data/training/bones_seed_v4_full_v1/train.csv`，计数包含镜像片段：

| 名称关键词 | 原始片段数 | 最终训练库保留数 |
| --- | ---: | ---: |
| `push_up` | 72 | 72 |
| `crawl` | 1874 | 1869 |
| `inchworm` | 24 | 22 |
| `elbow_plank` | 24 | 24 |
| `handstand` | 162 | 0 |

5条爬行因附录E穿地规则剔除，2条尺蠖因抬高支撑规则剔除，倒立类因关键词规则剔除。名称统计不是接触标注：爬行可能用手掌或前臂，肘撑平板不是手掌支撑。本模型每侧有腕roll，但没有独立腕pitch/yaw；运动学筛选通过不代表动态支撑可行。额外23DoF质量检查仍仅作诊断，不据此自动删除动作或改变当前训练库。

后续每个相关失败案例应记录：

1. 动作完整ID、源BVH与重定向版本、checkpoint迭代数、模型和评测配置、失败帧及原因。
2. 按同一50Hz时间轴对齐原始人体（注明缩放与坐标变换）、重定向参考及策略仿真轨迹，查看失败前后的叠加或并排视频。
3. 检查参考的手端位置/朝向、肩肘腕限位、身体姿态、穿地与支撑接触；检查仿真中的手/前臂碰撞几何、实际接触、滑移、关节跟踪和力矩限制。
4. 参考存在明显接触或姿态问题时，记录重定向问题证据；参考看似合理但策略偏离时，再检查控制和世界模型误差。仅靠运动学回放不能证明动态可行，无法区分时标为“待定”，允许混合原因。
5. 必要时在独立对照实验中修改重定向或控制配置，固定动作、初始条件和评测协议比较；证据充分后再决定修正数据或调整训练。

这里的执行效果首先指MuJoCo Warp仿真；本条记录不代表已验证真机上的手部承重能力。

核查日期：2026-10-27。复现仓库的成绩是作者报告，本工程没有重新训练其完整数据集验证成绩。

| 项目 | 实际数据路径 | 对本工程的帮助 |
| --- | --- | --- |
| Kang / Cuboi SuperMimic | 主要接收已有G1机器人CSV，转50Hz、算FK/速度、建参考库；Kang未公开内部datav5完整转换/划分 | 采用其世界模型、BPTT、紧凑状态和可微FK基础；不能据此还原原训练数据 |
| MorphMimic | G1路径读取BONES-SEED的G1数据；X2/T2路径提供NVIDIA SOMA Retargeter的机器人适配、批处理和筛选工具 | 参考按机器人做映射、限位检查、子集管理；不是“所有人体动作都先经GMR” |
| GMR | 人体BVH/SMPL-X → Mink两阶段IK → 目标机器人qpos | 本工程注册原生23模型，替换腕端身体名，再按原始采样时间输出 |

源码依据：

- [Kang DATA.md](https://github.com/kangzejian1896/supermimic-g1-reproduction/blob/5daf8586f26da3ef3bd99bdd875c8f8ce1b4e18e/docs/DATA.md)：CSV契约、真实FPS、未公开的内部转换/划分。
- [Kang motion_convert.py](https://github.com/kangzejian1896/supermimic-g1-reproduction/blob/5daf8586f26da3ef3bd99bdd875c8f8ce1b4e18e/src/supermimic/data/motion_convert.py)：按秒插值、四元数SLERP、MuJoCo FK。其重采样主思路正确，本工程增加时间戳验证、原生23映射和边界约束。
- [MorphMimic SOMA工具](https://github.com/lzyang2000/morphmimic/tree/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/tools/soma_retargeter)：`assemble_robots.py`、`derive_t2.py`、`stage_subset.py`、`limit_stats.py`。
- [GMR retarget核心](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/motion_retarget.py)：ConfigurationLimit、可选VelocityLimit、逐帧IK。

## 核查中遇到的具体问题

GMR默认 `unitree_g1` 指向29自由度模型，不会因目标项目叫23DoF就自动变成23。本工程注册新机器人键，XML与训练/FK同源，不走29维padding。

GMR [bvh_to_robot.py](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/scripts/bvh_to_robot.py) 的该版本在非循环模式末帧执行 `break` 后才有保存append，导致最后一帧未保存；SMPL-X示例在取帧前递增索引，会跳过首帧。新适配器直接遍历所有源帧，先保存结果，没有这些边界路径。

GMR的BVH示例接受默认30FPS参数，没有强制取BVH的 `Frame Time`；其 [SMPL-X快速重采样辅助函数](https://github.com/YanjieZe/GMR/blob/25f56db79ac411258b377b6d4ccba0c8ff3372ac/general_motion_retargeting/utils/smpl.py) 结合整数frame_skip、linspace和重新估计FPS，不能当作我们严格50Hz时间网格的依据。本工程在原采样率做IK，原时间戳一并保存，之后只重采样一次。

GMR内存qpos四元数wxyz，示例导出pickle的root_rot是xyzw。不能把两者混用。我们的NPZ记录quaternion_order；旧pickle必须显式声明顺序，且仅导入可信文件。

GMR的LAFAN1 loader有1.75m默认身高；可以用 `--human-height` 覆盖，实际采用值会写报告。原生23IK映射保留上游权重，替换缺失腕端身体名，并修正腕端原点差异：23模型的 `wrist_roll_rubber_hand` 相对29模型的 `wrist_yaw_link` 在零姿态局部x方向为−0.084m，不能只改名字。ConfigurationLimit开启；不把IK迭代中的速度限制误称为整段动作的动力学可行性保证。

## 23自由度数据的两条路径

**推荐：人体 → 23DoF IK。** 优化器在缺少腰roll/pitch、腕pitch/yaw时重新分配其他关节，以尽量满足身体目标。要检查腕姿态权重、躯干不可达朝向、脚滑、接触与限位。

**过渡：已有29DoF机器人轨迹 → 按名称投影到23。** 可以保留根和共有关节，再算23模型身体轨迹；不能还原“原29机器人的手脚空间目标”。必须经过质量报告和回放，适合初始工程测试，不等价于完整重定向。

运动学质量报告检查有限值、硬限位、速度/加速度、自碰撞穿透、足部碰撞几何到地面的距离。脚滑统计只是筛查线索，不能把摆动脚当作接触脚，也不能只凭脚踝原点高度判断穿地。最终需要训练后完整回放、固定条件评测来检验动态可跟踪性。

原始LAFAN1已下载，77段BVH已完成首次原生23转换及50Hz/FK检查，结果见 `data/retargeted/lafan1_native23_v1/manifest.json`。全部片段都触发了至少一项整段几何筛查，不应整库直接训练。尚需动作家族划分、接触/碰撞处理、片段级筛选和动态跟踪验收；不能声称已完成论文性能复现。

## 新查到的23自由度项目（2026-10-27）

必须核对关节清单。“23”可能是腿12+腰3+双臂8，也可能是本工程的腿12+腰1+双臂10。

| 项目 | 核查结果 | 可用部分及限制 |
| --- | --- | --- |
| [Clarence-Pfister/SOMA-Retargeter](https://github.com/Clarence-Pfister/SOMA-Retargeter) | `feature/g1-23dof-retargeting` 的CSV关节表与我们逐项一致 | 公开23目标、IK配置、脚部稳定器及自定义MJCF路径；输入是SOMA骨架，并非任意BVH |
| [Clarence-Pfister/GEM-X](https://github.com/Clarence-Pfister/GEM-X) | `integration/all` 的23模型保留waist_yaw及双腕roll | 视频→SOMA→23机器人；复用上述SOMA分支，不能算独立的一套重定向验证 |
| [haoziidiot-ctrl/Omniretarget_23dof](https://github.com/haoziidiot-ctrl/Omniretarget_23dof) | 源码存在 `g1_23dof`、`(lafan,g1_23dof)` 映射、单腰及腕roll约束 | Holosoma/OmniRetarget路线，提供脚部粘连和交互相关配置；本机尚未运行，脚部辅助link与我们模型不同；README有指向作者内部GitLab的训练/部署依赖 |
| [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP) | `g1_29dof_anneal_23dof` 为腿12+腰3+双臂8，双腕锁定 | 有完整SMPL→机器人流程，但不与本工程关节空间兼容 |
| [GMT官方仓库](https://github.com/zixuan417/humanoid-general-motion-tracking) | 当前公开XML同样是腰3、双臂8 | README仍写数据处理/retargeter将发布；不能当作已公开原生23重定向方案 |
| [awsdkk训练数据模板](https://github.com/awsdkk/Unitree_g1_23dof_Training_Template) | 有23机器人动作、策略及PKL→CSV→NPZ工具 | 可作数据格式/动作参考，未看到原始人体→23 IK完整流程 |

另查到 `wandering2025/PBHC_g1_SingleWaistYaw`：仓库名声称单腰，但读取的 `g1_23dof_lock_wrist_fitmotionONLY.xml` 仍含waist_roll/pitch、没有腕关节，不能仅凭名称确认兼容。

固定版本依据：

- [SOMA 23关节CSV表](https://github.com/Clarence-Pfister/SOMA-Retargeter/blob/6c0517236d98910a63a57b082c4a585b2776de91/soma_retargeter/assets/csv.py)
- [SOMA 23 IK参数](https://github.com/Clarence-Pfister/SOMA-Retargeter/blob/6c0517236d98910a63a57b082c4a585b2776de91/soma_retargeter/configs/unitree_g1/soma_to_g1_23dof_retargeter_config.json)
- [GEM-X模型](https://github.com/Clarence-Pfister/GEM-X/blob/3331a5635d15120d2555bb76d96a7d18dcb8b5dc/assets/robots/g1_23dof_holo_compat/g1_23dof_holo_compat.xml)
- [OmniRetarget LAFAN映射](https://github.com/haoziidiot-ctrl/Omniretarget_23dof/blob/ad698f37d4bd010d80f5a2428850eb788bca0a4a/src/holosoma_retargeting/config_types/data_type.py)
- [ASAP关节清单](https://github.com/LeCAR-Lab/ASAP/blob/df5320cc47dd8cad97961bdfabfe402dd62ad999/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml)
- [GMT模型](https://github.com/zixuan417/humanoid-general-motion-tracking/blob/2a590de25a1eb08e47491977a738549c22f16e1f/assets/robots/g1/g1.xml)

## SOMA本机试跑

2026-10-28新增：已按NVIDIA官方Robot Configurator教程，从本机原生23DoF URDF独立生成 `g1_23dof_urdf_v1`，并完成10段动作三组对照及接触修正消融。新配置没有采用社区标定参数；当前全量任务继续保留。配置来源、腕端定义、改善与退步、视频和运行入口见 [独立URDF配置与对照报告](../reports/SOMA_URDF_CONFIGURATION.md)。

已在 `<soma-root>` 安装上述23分支，使用独立Python3.12环境、Newton1.0.0和Warp1.12.0。离线重定向不更换训练的MuJoCo Warp后端。具体复现命令和质量结果见 [soma_trial.md](soma_trial.md)。

该分支的配置为IK24次、joint_limit_weight=10、smooth_joint_filter_weight=5.5；双脚位置/旋转权重30/2，骨盆30/2，双手2/1.2；feet stabilizer额外20次。**collision_weight=0，且执行代码未创建自碰撞目标**，所以不能把“有稳定器”理解成“无自碰撞”。

人体数据建议使用 [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) 的 `soma_uniform` BVH。该数据集由Bones Studio发布，包含142,220段（71,132原始+71,088镜像），约288小时、标称120Hz；另有保留演员比例的 `soma_proportional` 和预先重定向的G1 CSV。完整数据访问需要HF登录并接受数据许可。2026-10-28已确认HF访问权限，正在下载完整soma_uniform及元数据。按固定SONIC关键词规则，当前142,220条中有129,785条候选；加入SuperMimic附录E公开名称示例后剩129,647条（作者完整词表未公开），并已接入附录E几何筛选。下载后的23自由度筛选流程见 [批量准备说明](soma_trial.md#bones-seed批量准备)。
