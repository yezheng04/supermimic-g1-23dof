"""Run the frozen post-100k checkpoint comparison and log progress to W&B."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import wandb

from supermimic.data.packed_library import sha256

OUT = Path("reports/evaluation_post100k_20261031")


def save(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def paired(baseline, current, names):
    old={r['name']:r for r in baseline['clips']};new={r['name']:r for r in current['clips']}
    return dict(recovered=sum(old[n]['failed'] and not new[n]['failed'] for n in names),
        newly_failed=sum(not old[n]['failed'] and new[n]['failed'] for n in names),
        improved_return_over_001=sum(new[n]['norm_return']-old[n]['norm_return']>.01 for n in names),
        worsened_return_over_001=sum(new[n]['norm_return']-old[n]['norm_return']<-.01 for n in names),
        mean_return_delta=sum(new[n]['norm_return']-old[n]['norm_return'] for n in names)/len(names),
        note='0.01 is a descriptive return-difference threshold, not a significance test; per-clip failures may vary between repeats')


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    manifest=json.loads((OUT/'policies/manifest.json').read_text())
    for iteration in plan['iterations']:
        row=next(r for r in manifest if r['iteration']==iteration)
        assert sha256(OUT/'policies'/row['file'])==row['sha256']
    assert sha256(OUT/'library.pt')==selection['library_sha256']
    (OUT/'status.txt').write_text('RUNNING\n')
    reports=[]
    with wandb.init(project='supermimic-g1-23dof',entity='1406157281-',job_type='checkpoint_evaluation',
        name='post100k_vs80k_fixed_eval_20261031',dir=str(OUT),save_code=False,
        config={**plan,'source_training_run':'opv9t7zf','selection_sha256':sha256(OUT/'selection.json'),
                'group_counts':selection['group_counts'],'protocol':selection['protocol']},
        settings=wandb.Settings(disable_git=True,init_timeout=45)) as run:
        save(OUT/'wandb.json',dict(id=run.id,url=run.url))
        run.define_metric('checkpoint_iteration');run.define_metric('eval/*',step_metric='checkpoint_iteration')
        try:
            for iteration in plan['iterations']:
                path=OUT/f'eval_{iteration:07d}.json'
                if not path.exists():
                    with (OUT/f'eval_{iteration:07d}.log').open('w') as log:
                        subprocess.run([sys.executable,'-u','scripts/evaluate_checkpoint_series.py','one',
                            '--output',str(OUT),'--iteration',str(iteration),'--num-envs',str(plan['num_envs'])],
                            stdout=log,stderr=subprocess.STDOUT,check=True)
                report=json.loads(path.read_text())
                assert report['selection_sha256']==sha256(OUT/'selection.json') and report['num_envs']==plan['num_envs']
                assert len(report['clips'])==selection['unique_clips']
                reports.append(report)
                summary=[{k:v for k,v in r.items() if k!='clips'}|{'paired_uniform_vs80k':paired(reports[0],r,selection['groups']['uniform'])}
                         for r in reports]
                save(OUT/'comparison.json',summary)
                metrics={'checkpoint_iteration':iteration}
                for group,s in report['groups'].items():
                    for key in ['mean_norm_return','failed','failure_rate','mean_completion_fraction']:
                        metrics[f'eval/{group}/{key}']=s[key]
                metrics['eval/uniform/delta_return_vs80k']=summary[-1]['paired_uniform_vs80k']['mean_return_delta']
                run.log(metrics)
                s=report['groups']['uniform'];c=report['groups']['challenge_union']
                print(f"Completed {iteration}: uniform return={s['mean_norm_return']:.6f}, failed={s['failed']}/2000; challenge failed={c['failed']}/462",flush=True)
                save(OUT/'progress.json',dict(completed=[r['iteration'] for r in reports],total=len(plan['iterations'])))
            fields=['iteration','uniform_return','uniform_failed','challenge_return','challenge_failed','delta_return_vs80k','recovered','newly_failed']
            with (OUT/'comparison.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for r in reports:
                    u=r['groups']['uniform'];c=r['groups']['challenge_union'];p=paired(reports[0],r,selection['groups']['uniform'])
                    writer.writerow(dict(iteration=r['iteration'],uniform_return=u['mean_norm_return'],uniform_failed=u['failed'],
                        challenge_return=c['mean_norm_return'],challenge_failed=c['failed'],delta_return_vs80k=p['mean_return_delta'],
                        recovered=p['recovered'],newly_failed=p['newly_failed']))
            artifact=wandb.Artifact('post100k-vs80k-fixed-evaluation',type='evaluation')
            for f in [OUT/'selection.json',OUT/'plan.json',OUT/'comparison.json',OUT/'comparison.csv',*OUT.glob('eval_*.json')]:
                artifact.add_file(str(f))
            run.log_artifact(artifact)
            (OUT/'status.txt').write_text('COMPLETED\n')
        except BaseException:
            (OUT/'status.txt').write_text('FAILED\n')
            raise


if __name__=='__main__':main()
