# 正面镜头与浏览器 Demo

本版沿用白色 #ffffff、浅灰 #f5f5f7、正文 #1d1d1f、辅助灰 #6e6e73、交互蓝 #0071e3。标题 Inter Tight，正文 Inter/系统中文字体。

页面先展示成功动作，再进入可运行的浏览器舞台；左侧选择动作分类与具体片段，上方选择策略和视角，底部运行/暂停/重置。手机将选择器放到舞台上方。世界模型预录区保留并明确区分。

```
        正面 / 侧前方动作视频
        在浏览器里运行策略
  分类与动作 | 可旋转的 G1 三维舞台
       策略 / 运行 / 重置 / 视角
        世界模型预录对比
```

复核：沿用用户认可的视觉体系，不添加装饰性卡片；新增部分以实际控制和机器人画面为主体。移除俯卧撑、爬行以及翻转/侧滚失败视频。统计评测保持原始数值，精选演示不作为全库成功率。

官方网页确实包含独立 `/demo/`。其页面包含世界模型、目标、MuJoCo 开关；本项目独立实现自身 23DoF Actor 与 MuJoCo 浏览器仿真，不复制官方模型或受限制脚本。

浏览器目标：50Hz 控制，4×5ms 物理子步，750维历史观测，32维动作目标编码。目标编码由对应 checkpoint 的编码器预先计算；动作解码器在浏览器逐步读取实时仿真状态推理，不能播放预存动作充当策略。提供80k与104k策略。物理、输入排列、动作尺度、初始化与终止判定需要单独验证。

## 重新录制

先按 `docs/browser_demo.md` 筛选候选动作，再运行：

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python scripts/record_showcase.py \
  --selection reports/showcase_new/selection.json \
  --library reports/showcase_new/candidates.pt \
  --checkpoint models/models_0080000.pt --output reports/showcase_new/recording
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python scripts/render_showcase.py \
  --recording reports/showcase_new/recording --output reports/showcase_new/render
```

视频采用随根部位置和朝向平滑移动的侧前方镜头。世界模型镜头额外留出预测前进方向的空间；三种干预共用基线镜头，不改变机器人状态或人为移动预测残影。
