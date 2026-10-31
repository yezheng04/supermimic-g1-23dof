"""Validate and summarize the frozen 80k/128k flip diagnostics."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from supermimic.data.packed_library import sha256


def main():
    out = Path("reports/flip_diagnostics_128k_20261030")
    protocol = json.loads((out / "protocol.json").read_text())
    summaries, all_rows = [], []
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, iteration in zip(axes, (80000, 128000)):
        reports = [json.loads((out / f"eval_{iteration:07d}_r{i}.json").read_text()) for i in range(3)]
        for report in reports:
            assert report["protocol_sha256"] == sha256(out / "protocol.json")
            assert len(report["full_clips"]) == 54 and len(report["phases"]) == 10
            assert report["iteration"] == iteration
            for row in report["full_clips"] + report["phases"]:
                assert row["initial_state_max_abs_error"] < 1e-4
                assert row["steps"] <= row["remaining_steps"]
                if not row["failed"]:
                    assert row["steps"] == row["remaining_steps"]
                    assert row["root_height_error_m"] <= .25 + 1e-5
                    assert row["gravity_z_difference"] <= .8 + 1e-5
                else:
                    if "root_height" in row["failure_reason"]:
                        assert row["root_height_error_m"] > .25 - 1e-5
                    if "root_tilt" in row["failure_reason"]:
                        assert row["gravity_z_difference"] > .8 - 1e-5
                all_rows.append(row)
        returns = [r["summary"]["mean_norm_return"] for r in reports]
        phases = []
        for j, template in enumerate(reports[0]["phases"]):
            rows = [r["phases"][j] for r in reports]
            assert all(r["start_frame"] == template["start_frame"] for r in rows)
            end_times = [r["end_reference_seconds"] for r in rows]
            successes = sum(not r["failed"] for r in rows)
            summary = dict(start_seconds=template["start_seconds"], phase=template["phase"],
                successes=successes, repeats=3, end_seconds_min=min(end_times), end_seconds_max=max(end_times),
                mean_remaining_norm_return=float(np.mean([r["norm_return"] for r in rows])),
                initial_joint_clamp_max_deg=template["initial_joint_clamp_max_deg"],
                failure_reasons=sorted({r["failure_reason"] for r in rows if r["failed"]}))
            phases.append(summary)
            color = "#178c80" if successes == 3 else "#c74b45"
            start, end = template["start_seconds"], float(np.mean(end_times))
            ax.plot([start, end], [j, j], color=color, linewidth=5, solid_capstyle="round")
            ax.plot(start, j, "o", color="#334155", markersize=4)
            ax.plot(end, j, "o" if successes == 3 else "x", color=color, markersize=8)
            ax.text(7.07, j, f"{successes}/3", va="center", fontsize=10, color=color)
        changed = [name for name in [r["name"] for r in reports[0]["full_clips"]]
                   if len({next(r["failed"] for r in report["full_clips"] if r["name"]==name) for report in reports}) > 1]
        summaries.append(dict(iteration=iteration, repeats=3,
            full_clip_successes=[r["summary"]["successful"] for r in reports],
            full_clip_mean_norm_return=float(np.mean(returns)),
            full_clip_mean_norm_return_range=[min(returns), max(returns)],
            full_clip_failure_flag_changed_names=changed, phases=phases))
        ax.set(title=f"Checkpoint {iteration:,}", xlabel="Original reference time (s)", xlim=(-.15, 7.7))
        ax.axvline(6.94, color="#94a3b8", linestyle="--", linewidth=1)
        ax.grid(axis="x", color="#e2e8f0")
        ax.set_yticks(range(10), [f"Start {r['start_seconds']:.2f}s" for r in reports[0]["phases"]])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Flip 360: reference-state starts and suffix completion", fontsize=15)
    fig.text(.5, .015, "Red x: terminated | Teal: reached clip end | Count: successful repeats / 3\n"
             "Supplied reference pose and velocity; standard soft-limit clipping. Suffix success is not full-flip success.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(out / "phase_comparison.png", dpi=160)
    plt.close(fig)
    summary = dict(protocol_sha256=sha256(out / "protocol.json"), checkpoints=summaries,
        integrity_checks="Coverage, same protocol, reset pose/velocity parity after documented clipping, suffix lengths, pre-reset terminal thresholds all passed")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    keys = [k for k in all_rows[0] if k != "initial_joint_clamps"]
    with (out / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_rows)
    lines = ["# Flips 80k / 128k 固定复测与分阶段诊断", "", "日期：2026-10-30。训练任务未修改；全部测试在本机RTX 5080运行。", "",
        "54条沿用原评测集合（含镜像），从首帧到原始片段结束或论文失败条件。每个checkpoint独立重复3次，seed=0、关闭噪声和域随机化、摩擦1.0、50Hz；完整动作归一化回报分母覆盖整个片段。",
        "", "| checkpoint | 三次完整动作成功数 | 平均归一化回报（三次均值） |", "|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['iteration']:,} | {' / '.join(str(n) for n in s['full_clip_successes'])}，每次共54条 | {s['full_clip_mean_norm_return']:.4f} |")
    lines += ["", "## 同一条 Flip 360 的分阶段起点", "", "动作：`230531__flip_360_001__A415`，原始参考时长6.94秒。", "",
        "起点按参考运动学选定，名称不等于经接触检测确认的腾空/落地事件。初始化参考根姿态、线速度、角速度、关节位置和速度；沿用训练软限位裁剪、历史观测与上一动作重置。不是从头执行后抵达这些状态。",
        "", "| 起点 | 参考阶段 | 80k完成剩余片段 | 128k完成剩余片段 | 128k结束的原始时间 | 初始最大关节裁剪 |", "|---|---|---|---|---|---|"]
    for a,b in zip(summaries[0]["phases"],summaries[1]["phases"]):
        end = f"{b['end_seconds_min']:.2f}" if b['end_seconds_min']==b['end_seconds_max'] else f"{b['end_seconds_min']:.2f}–{b['end_seconds_max']:.2f}"
        lines.append(f"| {b['start_seconds']:.2f}s | {b['phase']} | {a['successes']}/3 | {b['successes']}/3 | {end}s | {b['initial_joint_clamp_max_deg']:.2f}° |")
    lines += ["", "## 结论与限制", "",
        "- 12.8万轮仍未通过任一完整flip；这54条的平均回报比8万轮下降。不能以迭代数增加代替性能验证。",
        "- 两个checkpoint的阶段成功模式一致：从0–3.14秒的选定起点均失败，从3.32秒及以后选定起点均可完成剩余动作。当前问题不只是起跳前的衔接，给定倒立附近参考状态后也未完成后续。",
        "- 后段成功不表示整个翻转成功，也不证明末段已精确跟踪；成功仅为没有触发高度/倾斜终止并到达片段末尾。",
        "- 初始化对部分关节执行既有0.9软限位裁剪，最大4.37°；已逐项保存，不能声称所有起点都与原参考关节角完全一致。原始发现保留在initialization_mismatch.json。",
        "- 三次成功/失败结论一致，但回报和个别终止时刻略变，未达到逐条数值严格确定性。",
        "- 仍不能把原因唯一归结为策略、世界模型或23DoF重定向。需要继续检查起跳/翻转阶段参考的接触和动力学可行性，以及该阶段世界模型预测误差。",
        "", "## 可追溯文件", "",
        "结果目录：`reports/flip_diagnostics_128k_20261030/`，包含protocol.json、checkpoint.json、summary.json、results.csv、6份原始评测JSON与轨迹NPZ。",
        "", "复测入口：", "", "```bash",
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python scripts/evaluate_flip_diagnostics.py evaluate \\",
        "  --checkpoint reports/flip_diagnostics_128k_20261030/policy_0128000.pt --repeat 3",
        "```", "", "默认汇总只读取编号0、1、2的冻结重复实验。新增重复使用新编号，避免覆盖。", ""]
    Path("docs/flip_diagnostics_128k.md").write_text("\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
