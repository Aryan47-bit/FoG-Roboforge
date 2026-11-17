import json
from pathlib import Path
import numpy as np
from data_pipeline import ROOT,OUT

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit=json.loads((ROOT/'reports/dataset_audit.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,d in zip(axes,['daphnet','fogstar']):
        ids=sorted({r['participant'] for r in audit if r['dataset']==d})
        # Deduplicate bilateral observations in the duration plot.
        rows=[r for r in audit if r['dataset']==d and r['side']!='R']
        fog=[sum(r['fog_seconds'] for r in rows if r['participant']==p)/60 for p in ids]
        normal=[sum(r['normal_seconds'] for r in rows if r['participant']==p)/60 for p in ids]
        ax.bar(range(len(ids)),normal,label='Non-FoG');ax.bar(range(len(ids)),fog,bottom=normal,label='FoG')
        ax.set_title(d);ax.set_ylabel('Minutes');ax.set_xticks(range(len(ids)),[p.split(':')[1] for p in ids],rotation=90);ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'reports/label_distribution.png');plt.close(fig)
    records=json.loads((OUT/'manifest.json').read_text());fig,axes=plt.subplots(2,1,figsize=(12,6))
    for ax,d in zip(axes,['daphnet','fogstar']):
        for r in records:
            if r['dataset']!=d:continue
            z=np.load(ROOT/r['file']);ids=np.flatnonzero(z['labels']==1)
            if not len(ids):continue
            a=max(0,int(ids[0])-200);b=min(len(z['t']),a+1500);t=z['t'][a:b]-z['t'][a]
            ax.plot(t,z['x'][a:b]);ax.fill_between(t,-20,20,where=z['labels'][a:b]==1,alpha=.15,color='red')
            ax.set_title(d+' '+r['recording']);ax.set_ylabel('m/s²');ax.set_xlabel('Seconds');break
    fig.tight_layout();fig.savefig(ROOT/'reports/example_signals.png');plt.close(fig)
    results_path=ROOT/'reports/experiment_summary.json'
    if not results_path.exists():return
    results=json.loads(results_path.read_text());selected=json.loads((OUT/'selected.json').read_text())
    lines=['# Training and evaluation report','',f'Selected using validation only: **{selected["experiment"]}**.','',
           '| Experiment | Test PR-AUC | Event sensitivity | False events/hour | Median delay (s) |',
           '|---|---:|---:|---:|---:|']
    for r in results:
        e=r['test']['event'];w=r['test']['window']
        lines.append(f'| {r["name"]} | {w["pr_auc"]} | {e["sensitivity"]:.3f} | {e["false_alarms_per_hour"]:.2f} | {e["median_delay_s"]} |')
    chosen=next(r for r in results if r['name']==selected['experiment'])
    lines+=['','## Combining datasets','',
            'The following comparison uses validation-selected individual models and the validation-selected combined model, evaluated on the same frozen participants within each dataset. Event metrics share ground truth even when window-overlap labels differ.','']
    for dataset in ['daphnet','fogstar']:
        individual=max([r for r in results if r['name'].startswith(dataset+'_')],key=lambda r:r['validation_selection_score'])
        a=individual['test']['groups']['dataset:'+dataset]['event']
        b=chosen['test']['groups']['dataset:'+dataset]['event']
        lines.append(f'- {dataset}: individual `{individual["name"]}` sensitivity {a["sensitivity"]:.3f}, false events/hour {a["false_alarms_per_hour"]:.2f}; combined sensitivity {b["sensitivity"]:.3f}, false events/hour {b["false_alarms_per_hour"]:.2f}.')
    lines+=['','## Either-side observations','']
    for side in ['L','R']:
        e=chosen['test']['groups']['side:'+side]['event']
        lines.append(f'- {side} ankle, held-out participants: sensitivity {e["sensitivity"]:.3f}, false events/hour {e["false_alarms_per_hour"]:.2f}.')
    deployment=OUT/'deployment/validation.json'
    if deployment.exists():
        q=json.loads(deployment.read_text());e=q['test_int8']['event'];f=q['test_float']['event']
        lines+=['','## Export and quantization','',
            f'Float TFLite: {q["float_bytes"]:,} bytes. Full INT8 TFLite: {q["int8_bytes"]:,} bytes. Input/output are INT8; input shape is 1 × 100 × 3.',
            f'PyTorch/Keras maximum absolute probability difference: {q["torch_keras_max_absolute_error"]:.3g}.',
            f'INT8 held-out event sensitivity: {e["sensitivity"]:.3f}; false events/hour: {e["false_alarms_per_hour"]:.2f}; median delay: {e["median_delay_s"]:.2f} s.',
            f'Quantization event-sensitivity change: {100*(e["sensitivity"]-f["sensitivity"]):.2f} percentage points. Maximum calibration-window probability difference: {q["calibration_max_probability_error"]:.3f}.',
            'All exported operator names are covered by the candidate firmware resolver. This is a static operator check, not a TFLM compile or hardware compatibility result.',
            'Export artifacts: `artifacts/deployment/`. Model header and normalization/decision config are also placed beside the firmware sketch.']
    lines+=['','## Limits','']
    lines+=['','These are sensor-stream metrics. Bilateral FoG-STAR observations are correlated; pooled event counts do not represent unique clinical episodes.',
            'Matching is chronological, one-to-one, strictly positive overlap after the first available window-end decision. Duplicate predicted events count as false events. No tolerance or backdating is applied.',
            'False-event rate divides by valid observed sensor time, including FoG; warm-up and missing intervals are excluded. Full-recording sensitivity, including unobservable episodes, has not been established.',
            'Small held-out groups have high uncertainty. Native axes remain dataset-specific. Either-ankle conclusions apply only to these recorded placements.',
            'Training used the RTX 4050. Public-dataset evaluation does not establish clinical performance or arbitrary mounting tolerance.',
            'The full multimodal annotated release is missing. Three-source comparisons and that dataset baseline are blocked.',
            'Board memory, latency, firmware compilation, hardware-in-the-loop parity, and long-duration stability have not been measured. This is not deployment-ready.']
    (ROOT/'reports/RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for r in results:
        h=json.loads((OUT/'experiments'/r['name']/'history.json').read_text())
        axes[0].plot([v['epoch'] for v in h],[v['loss'] for v in h],label=r['name'])
        axes[1].plot([v['epoch'] for v in h],[v['validation_pr_auc'] for v in h])
    axes[0].set_title('Training loss');axes[1].set_title('Validation PR-AUC');axes[0].legend(fontsize=5)
    fig.tight_layout();fig.savefig(ROOT/'reports/training_curves.png');plt.close(fig)

if __name__=='__main__':main()
