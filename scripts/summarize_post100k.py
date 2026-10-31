"""Summarize the fixed post-100k comparison, separating coverage and return."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=Path('reports/evaluation_post100k_20261031')
    plan=json.loads((out/'plan.json').read_text())
    reports=[json.loads((out/f'eval_{it:07d}.json').read_text()) for it in plan['iterations']]
    baseline=reports[0]['groups']['uniform']
    x=[r['iteration']/1000 for r in reports]
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for ax,group,key,title in [(axs[0],'uniform','mean_norm_return','Fixed 2,000 motions: full-duration normalized return'),
                         (axs[1],'uniform','failed','Failed motions / 2,000'),
                         (axs[2],'challenge_union','failed','Diagnostic challenge group: failed motions / 462')]:
        y=[r['groups'][group][key] for r in reports]
        ax.plot(x,y,'o-',color='#287f9b');ax.axhline(reports[0]['groups'][group][key],color='#b86a36',linestyle='--',label='80k baseline')
        ax.set_title(title);ax.grid(alpha=.2);ax.legend()
        for a,b in zip(x,y):ax.annotate(f'{b:.3f}' if key=='mean_norm_return' else str(b),(a,b),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
        ax.margins(y=.2)
    axs[2].set_xlabel('Checkpoint iteration (thousands)')
    fig.tight_layout();fig.savefig(out/'checkpoint_comparison.png',dpi=160);plt.close(fig)
    lines=['# 10万轮之后checkpoint与8万轮的固定评测对比','',
        '日期：2026-10-31。远端正式训练已因Critic非有限梯度停止，本次没有恢复或修改训练。',
        '', '## 协议', '',
        '沿用此前冻结的2,455条动作、同一缓存和50Hz时间轴。2,000条等概率主样本用于整体比较，462条困难组并集单列，去重合计2,455条；包含镜像片段。这是训练集覆盖评测，不是未见动作泛化测试，也不代表全库129,477条的最终结果。',
        '', '本机RTX5080、256环境、seed=0，关闭噪声和域随机化、摩擦1.0，从首帧执行到片段末尾或原始失败条件。归一化回报按完整参考时长计算，提前终止的剩余部分计零。每个模型源文件和策略副本都检查哈希、迭代号和有限值。',
        '', '8万轮在本次重新评测。已知GPU物理存在逐片段重复差异，接近的结果不能只凭一次运行断言显著改善。',
        '', '## 主样本', '', '| checkpoint | 完整回报 | 相比80k | 失败/2000 | 困难组失败/462 |', '|---|---|---|---|---|']
    for r in reports:
        s=r['groups']['uniform'];c=r['groups']['challenge_union']
        lines.append(f"| {r['iteration']:,} | {s['mean_norm_return']:.4f} | {s['mean_norm_return']-baseline['mean_norm_return']:+.4f} | {s['failed']} | {c['failed']} |")
    groups=['push_up','crawl','inchworm','elbow_plank','flip','side_roll','broad_jump','lying_recovery','long_duration']
    lines+=['','## 困难动作组失败数','','| checkpoint | '+' | '.join(groups)+' |','|'+'---|'*(len(groups)+1)]
    for r in reports:lines.append('| '+str(r['iteration'])+' | '+' | '.join(f"{r['groups'][g]['failed']}/{r['groups'][g]['clips']}" for g in groups)+' |')
    lines+=['','组名来自动作名称关键词，不等于官方接触标签。组间存在重叠；诊断组平均值不能与主样本直接混为全库成绩。',
        '', '## 文件与运行', '',
        '- `reports/evaluation_post100k_20261031/comparison.json`：每个检查点的分组结果及相对80k逐动作变化。',
        '- `comparison.csv`：主指标摘要。`eval_*.json`：全部逐片段回报、失败时刻和失败原因。',
        '- `plan.json`、`selection.json`、`policies/manifest.json`：固定实验清单、模型来源和哈希。',
        '', '```bash', 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python scripts/evaluate_post100k.py',
        '.venv/bin/python scripts/summarize_post100k.py','```','']
    wb=json.loads((out/'wandb.json').read_text());lines.append(f"W&B：{wb['url']}")
    Path('docs/checkpoint_evaluation_post100k.md').write_text('\n'.join(lines)+'\n')
    print((out/'comparison.csv').read_text())


if __name__=='__main__':main()
