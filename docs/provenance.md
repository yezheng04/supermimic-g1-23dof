# 来源与版本

- 训练基础：[kangzejian1896/supermimic-g1-reproduction](https://github.com/kangzejian1896/supermimic-g1-reproduction)，commit `5daf8586f26da3ef3bd99bdd875c8f8ce1b4e18e`，其上游为CuboiLeo复现。本项目拷贝并适配了`src/supermimic`和若干数学测试，MIT许可证保留在 `licenses/supermimic-upstream-MIT.txt`。
- 原生机器人：本机 `unitree_rl_mjlab/src/assets/robots/unitree_g1` 的23DoF MJCF、所需网格和23DoF配置，仓库HEAD `1425b15f73bd4095f0df53709d7c389c3eb9e790`。使用本机文件快照，不宣称与远端HEAD文件完全相同。许可证见 `licenses/unitree_rl_mjlab-LICENCE.txt`；修改资源路径使其独立加载。
- GMR：[YanjieZe/GMR](https://github.com/YanjieZe/GMR)，commit `25f56db79ac411258b377b6d4ccba0c8ff3372ac`；两份IK映射源自其JSON，修改腕端名称。MIT许可证见 `licenses/GMR-MIT.txt`。该路径仅属历史记录，当前已移除 GMR 集成；保留来源及许可供追溯。
- 设计参考：[lzyang2000/morphmimic](https://github.com/lzyang2000/morphmimic)，commit `aa40fe46a7fa995ba7a7354ad065a2c8ad64a302`。仅核查设计和数据流程，未将其多机器人网络作为本工程实现。
- 理论依据：[SuperMimic论文](https://arxiv.org/pdf/2610.28480)。论文约束与推断实现的区别记录于 `architecture.md`。
- RTX5080工程参考：[rzc156/supermimic-rtx5080](https://github.com/rzc156/supermimic-rtx5080)，commit `df967c61c29a2ed9ba6b188e1607ba59225fe8a7`。核查其配置、编译入口及公开评测结果，未复制训练实现或更改正式训练规模；详见 `../evidence/RTX5080_REFERENCE_AUDIT.md`。
- 双卡工程设计直接核查：[CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)，2026-10-29 HEAD `ba6105349069bf040f1fdad60873d13138d505a2`。本地分布式辅助模块与该版本相同。其e8采用每卡Actor批量2048；本项目双96GB配置保持论文每卡4096，具体未知实现仍不能视为原作者官方代码。

本工程新增相位一致的mjlab1.2环境适配、源时间戳转换、原生23自由度SOMA适配、数据验证、CLI和回归测试。未修改共享环境的mjlab或原用户仓库。未复制上游数据集、私有manifest或已有训练权重。`tests/fixtures/synthetic_lafan.bvh`为程序生成的测试骨架；站立示例由机器人HOME姿态生成。

- 2026-10-29进一步按上述CuboiLeo版本核查全部41个源模块，并适配其`diag_drift.py`、`play.py`和`eval_push.py`设计。新增本地有界并发fp16打包、manifest/模型验证、packed评测入口与Viewer接口；改动范围和保留差异见`cuboileo_adaptation.md`，MIT来源与许可证保持保留。

- 当前重定向使用 [NVIDIA SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) 及独立的社区23DoF适配路径；版本和具体用途见 `soma_trial.md`，Apache-2.0 许可见 `licenses/SOMA-Retargeter-Apache-2.0.txt`。
- 展示页仅参考 SuperMimic / Proteus 的信息组织与留白风格，未拷贝其视频、成绩或站点源码。
