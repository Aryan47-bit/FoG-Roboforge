"""Predeclared second development round. Existing test set is now a reused benchmark."""
import json
import torch
from data_pipeline import ROOT,OUT,save_json
from train import train_one

def main():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    records=json.loads((OUT/'manifest.json').read_text())
    protocol={
        'changes':'24-channel causal residual TCN; receptive field 61 samples instead of 13 for baseline CNN',
        'experiments':[['gravity',.2],['dynamic',.2],['dynamic',.5]],
        'epochs_max':40,'selection':'validation event F1 - 0.002 * false alarms/hour',
        'engineering_target_only':{'sensitivity_min':.9,'false_events_per_hour_max':2,'median_delay_s_max':2},
        'test_caveat':'Previously reported test set is a reused benchmark, not fresh independent certification. No thresholds are fit on it.',
        'original_deployment_artifact':'preserved; no automatic replacement'}
    save_json(ROOT/'reports/improvement_protocol.json',protocol)
    results=[]
    for representation,overlap in protocol['experiments']:
        name=f'v2_longtcn_{representation}_{overlap}'
        results.append(train_one(name,records,'longtcn',40,representation,overlap))
    baseline=json.loads((OUT/'experiments/combined_cnn_gravity_0.2/result.json').read_text())
    best=max([baseline]+results,key=lambda r:r['validation_selection_score'])
    event=best['test']['event']
    meets=event['sensitivity']>=.9 and event['false_alarms_per_hour']<=2 and event['median_delay_s'] is not None and event['median_delay_s']<=2
    save_json(ROOT/'reports/improvement_results.json',{'protocol':protocol,'selected':best['name'],'results':results,'baseline':baseline,'engineering_target_met_on_reused_benchmark':meets,'deployment_ready':False})
    lines=['# Second development round','',protocol['test_caveat'],'',
        '| Model | Validation score | Benchmark sensitivity | False events/hour | Median delay |',
        '|---|---:|---:|---:|---:|']
    for r in [baseline]+results:
        e=r['test']['event'];lines.append(f'| {r["name"]} | {r["validation_selection_score"]:.3f} | {e["sensitivity"]:.3f} | {e["false_alarms_per_hour"]:.2f} | {e["median_delay_s"]} |')
    lines+=['',f'Validation-selected: {best["name"]}. Engineering target met: {meets}.',
        'The target of 90% sensitivity, <=2 false events/hour and <=2 s median delay is provisional, not a clinical acceptance criterion.',
        'Existing deployment artifacts are preserved. Fresh participant data, source/device filter parity, board compilation, resource measurements and hardware validation remain necessary.']
    (ROOT/'reports/IMPROVEMENT.md').write_text('\n'.join(lines))
    print('SELECTED',best['name'],'TARGET_MET',meets,flush=True)

if __name__=='__main__':main()
