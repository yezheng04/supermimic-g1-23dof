# 动作与模型资产

## 查看已有演示

```bash
python3 scripts/download_demo_assets.py          # 四个 MP4，约 39 MB
python3 scripts/download_demo_assets.py --models # 额外下载 80k Actor / WM，约 68 MB
```

下载使用当前 `gh auth` 登录态访问私有 Release；不在源码或网页中保存凭据。每个文件在写入目标路径前检查大小与 SHA-256。海报和评测摘要已在 Git 中，不需要额外下载。

模型包 `models/models_0080000.pt` 包含 `policy`、`world_model`、`config` 和迭代信息，缺少 Critic、优化器、Replay 和 RNG，不能作为完整续训 checkpoint。

## 准备训练数据

当前训练使用 129,477 条原生 23DoF 动作、47,329,219 帧，约 262.22 小时。原始 BONES-SEED、重定向 NPZ 与约 5.2GiB 的 fp16 缓存不在 Git 或本次 Release 中。应从数据提供方取得数据并遵守其许可，不从本展示页下载原始人体数据。

已有完整缓存时，保持以下相对路径即可使用双卡配置：

```text
data/training/bones_seed_v4_full_v1/train.csv
data/training/bones_seed_v4_full_v1/library_native23_fp16.pt
```

缓存加载会验证 manifest、原生机器人模型和关节顺序；不要直接替换成 29DoF 缓存。

重新生成时的流程：BONES-SEED `soma_uniform` BVH → SOMA 原生 23DoF IK → 源时间轴插值到 50Hz → 重新计算 FK / 速度 → 附录 E 筛选 → manifest / fp16 打包。额外原生 23DoF 质量检查仅作诊断。

```bash
# 在 SOMA 的独立 Python 3.12 环境下使用对应版本入口。
../SOMA-Retargeter/.venv/bin/python scripts/prepare_bones_seed_batch.py --help
.venv/bin/sm23 pack data/retargeted/bones_seed_native23_v4/library_50hz \
  data/training/bones_seed_v4_full_v1/library_native23_fp16.pt \
  --manifest data/training/bones_seed_v4_full_v1/train.csv --workers 8
```

全量训练采用社区 SOMA 23DoF 路径；自建 URDF 标定是独立对照，没有替换该次训练数据。版本、参数及入口见 [SOMA 流程记录](soma_trial.md) 和 [重定向核查](retargeting_audit.md)。`assets/robotics/unitree/g1_23dof_urdf_v1/` 保留自建目标模型与配置；可用 `scripts/retarget_soma_official.py` 单独试验。两条 SOMA 路径的 API 不相同，不能互换 Python 环境。

## 网站素材含义

| 视频 | 蓝色内容 | 对齐方式 |
| --- | --- | --- |
| `future-trail.mp4` | 世界模型未来姿态 | 每帧重新预测 32 步，预测状态内闭环计算动作 |
| `world-model.mp4` | 同一时刻的世界模型预测 | 使用实际已记录动作，比较 32 步窗口与整段连续预测 |
| `side-roll.mp4`、`flips.mp4` | 重定向参考机器人 | 同时间、同空间，无逐帧对齐修正；失败后策略冻结 |

视频全部为 80k checkpoint 仿真结果。预录视频的播放调速仅改变观看速度，不改变训练或数据时间轴。
