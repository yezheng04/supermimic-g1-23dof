# rzc156/supermimic-rtx5080 核查（2026-10-29）

核查版本：[df967c61c29a2ed9ba6b188e1607ba59225fe8a7](https://github.com/rzc156/supermimic-rtx5080/tree/df967c61c29a2ed9ba6b188e1607ba59225fe8a7)。只读取源码、配置和公开结果，没有运行外部安装/训练脚本。正式 `bones_full_v1.json` 保持论文规模与当前观测设置。

## 主要结论

值得参考的是RTX5080实测编译/吞吐，以及修正后的评测记录。该仓库基于CuboiLeo复现，我们经Kang分支也来自这一上游，因此相似网络不是独立验证原文未知参数的证据。

公开的5080主结果使用29DoF、LAFAN1 40条、2048环境、WM批量2048、Actor批量2048、1000步Replay、H=32、UTD=0.4，开启WM与ImagStep的torch.compile，并加入9维锚点相对位姿到goal。它不是5080上4096/4096加BONES全库的成功案例。README中BONES全库的结果来自继承的5090上游实验。

## 源码与配置对照

| 项目 | 外部5080公开运行 | 我们正式入口 |
| --- | --- | --- |
| 机器人 | 29DoF | 原生23DoF |
| 数据 | LAFAN1 40条 | 全部129,477条BONES保留动作 |
| 环境/Actor批量 | 2048/2048 | 4096/4096 |
| WM | 10批×2048、8步窗口 | 相同 |
| Replay | 1000步/环境 | 相同 |
| 编译 | 两处default | 两处关闭 |
| goal锚点反馈 | 开启，额外9维 | 关闭 |
| 主网络宽度/latent/终止头 | 与我们的对应网络一致 | 23DoF适配输入输出维度 |

逐文件比较：`imagination/compiled.py`与本项目完全相同，已有编译入口，无须复制一套新实现。`policy_trainer.py`主算法相同，我们另外保留非有限梯度报错和clip末尾无效Critic样本屏蔽。其终止头也采用4维误差特征、64/64隐藏层；状态损失尺度、增量积分、SiLU及Actor损失除以H等也与当前工程一致，但这些仍是共同上游的实现选择，不是新增的论文依据。

上游还有不能直接覆盖的差异：默认奖励锚点为torso_link，COM随机化为startup；我们正式模式采用pelvis及按回合重采样。依赖环境版本也不同，不能直接覆盖本项目虚拟环境。

## 显存与结果证据

作者报告开启compile后2048/2048的峰值约11.3GB，并报告4096/4096 OOM。这与本机此前未编译论文规模的OOM方向一致，但不构成本机编译模式的实际测试。

公开评测JSON及summary.csv确认：40k单次平均回报0.8514849、失败2/40；48k三次平均约0.8375、失败4–6/40。仓库About仍写0.874，不能据此替代修正后结果。结果来自额外锚点反馈和单训练seed，作者自己的纯本体对照尚未跑，不能将上游5090的对照当成本机5080实测。

文档关于论文观测/评测缺项的解释是作者假设。Table A.1只列本体观测，而未来goal字段并未完整公开，因此不能据其结果断言论文没有任何相对位姿反馈、论文必须采用分段评测，或所有复现都必须开启anchor_obs。

## 对本工程的可用改进

1. 优先隔离测试现有torch.compile入口，保持4096环境/4096有效Actor批量、H=32和现有观测，再看本机性能与显存。外部11.3GB结果不能直接外推我们的全库。
2. 若仍OOM，研究微批量累积与激活重计算。外部源码未提供相应解决方案；不能把其2048/2048称为保留论文有效批量。
3. 将anchor相对反馈作为独立消融候选，主配置不自动开启。23DoF下goal由560变569维，会改变网络输入；需要独立初始化/实验口径。
4. 保持长期学习率衰减时间轴，按训练步数比较多个检查点。我们延长pilot已经这样处理，但850步小集诊断不能与40k/48k成熟结果比较。
5. 同时报告全片段分母回报、失败数和重复评测波动。本项目已有全长分母、N−1步、固定条件与失败标签；无需回退现有对齐实现。

## 直接来源

- [实际运行配置](https://github.com/rzc156/supermimic-rtx5080/blob/df967c61c29a2ed9ba6b188e1607ba59225fe8a7/results/lafan1_anchor/config.json)
- [显存与结果报告](https://github.com/rzc156/supermimic-rtx5080/blob/df967c61c29a2ed9ba6b188e1607ba59225fe8a7/REPRODUCTION.md)
- [训练启动脚本](https://github.com/rzc156/supermimic-rtx5080/blob/df967c61c29a2ed9ba6b188e1607ba59225fe8a7/scripts/run_lafan1.sh)
- [修正后结果](https://github.com/rzc156/supermimic-rtx5080/blob/df967c61c29a2ed9ba6b188e1607ba59225fe8a7/results/lafan1_anchor/summary.csv)
