# 8万次检查点：仿真执行与世界模型预测叠加

用户进一步指定官网展示模式后，新增主展示版：

`reports/demo_80k_future_trail_20261030/g1_23dof_80k_future_trail.mp4`

新版采用单画面白色背景，实体机器人与8个渐淡蓝色未来姿态同坐标叠加。每个视频时刻从当前仿真状态和过去10帧观测出发，在世界模型内执行32步策略闭环预测；每步动作由预测状态历史重新计算，展示+0.08、+0.16、…、+0.64秒的姿态。蓝色轨迹不是过去残影、不是参考动作，也不是同一时刻的误差对比。预测不读取未来实际状态或未来实际动作，片段结束后隐藏超出参考长度的虚影。

展示样式参考[官网World-model prediction](https://supermimic.github.io/#closer)的Policy on模式。0.64秒时域和8个采样间隔是本工程明确选择，并非声称取得了官方绘制参数；当前只制作Policy on展示，未实现官网Policy off或Slippery floor实验。

新版复用下文8段已录制的真实仿真轨迹，基于解析观测重建历史后，起始动作与记录动作的最大绝对差为0.002395、RMSE为0.000259（归一化动作单位），记录于`prediction.json`。差异未超过本次0.01检查阈值；不声称逐位一致。`tests/test_world_model_demo.py`另外验证策略动作确实随预测状态逐步更新。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/record_future_trail_demo.py
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/render_future_trail_demo.py
```

以下保留第一版左右分屏诊断视频的说明，两版预测协议不同，不应混用解读。

使用远端 `bones_full_2gpu/ckpt_0080000.pt` 中同一时刻的策略和世界模型权重，在本机RTX5080录制。主视频：

`reports/demo_80k_world_model_20261030/g1_23dof_80k_sim_vs_world_model.mp4`

灰色实体是策略驱动的MuJoCo Warp仿真机器人，蓝色虚影是世界模型预测；不是实体机器人拍摄，也不是参考动作。CuboiLeo首页原演示的虚影是参考动作，本视频仅借鉴叠加呈现形式，预测来源不同。

## 两种预测方式

- 左侧：32步开环预测，每32个控制步（0.64秒）从仿真状态重新初始化；窗口内递归使用模型自己的预测状态，不逐帧纠正。
- 右侧：仅动作首帧从仿真状态初始化，整段递归预测，不再读取仿真状态。相机根据两者间距自动拉远，机器人与预测轨迹本身没有任何逐帧平移、朝向或高度对齐。
- 两种预测都输入同一组仿真实际执行的动作和仿真上下文。这是相同控制输入下的动力学预测比较；不是让策略在预测状态上另行闭环控制。
- 已关闭动作、观测、重置噪声及域随机化，足部摩擦1.0，原生23DoF。控制与数据50Hz，视频每两帧取一帧输出25fps，按原始时间播放。
- 仿真失败帧在自动重置前捕获，视频停止于失败或动作末尾，不把重置姿态当成连续运动。数值无效的预测会标记，不通过静默重置恢复。

## 动作与解释

共8段：行走、慢跑、跳跃、街舞、俯卧撑、爬行、尺蠖动作及另一段尺蠖失败诊断。动作ID、完整参考长度、实际执行长度、终止原因及本次录制的完整回报保存在 `recording.json`。这些是演示案例，不代表整体成功率；本次8环境录制的轨迹不保证与之前256环境评测逐帧相同。

32步窗口内的平均根位置误差在这8段中约0.63–4.54厘米；整段连续预测会累积明显误差，例如行走片段末尾根位置相差约2.95米。视频如实展示漂移，不能从左侧重叠较好推断整段世界模型准确，也不能仅凭本视频判断重定向是否正确。

## 文件与复现

- `models_0080000.pt`：策略与世界模型评测副本，无优化器，不能用于续训；源和副本校验值在 `checkpoint.json`。
- `motions.pt`：从既有训练缓存原样切出的8条动作，未重新重定向或重采样。
- `trajectories.npz`：仿真状态、执行动作、上下文、两类模型预测、误差及终止信息。
- `clips/`：逐动作视频；`previews/`：预览帧；`video.json`：编码与播放信息。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/record_world_model_demo.py
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/render_world_model_demo.py
```

`tests/test_world_model_demo.py`验证连续预测不偷偷读取未来仿真状态、窗口只在指定边界重置，以及失败帧在自动重置前保存。视频由模型和轨迹程序渲染，无生成式画面。
