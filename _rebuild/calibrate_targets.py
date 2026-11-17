"""Validation-only feasibility sweep; never changes deployed artifacts."""
import json
import numpy as np
from data_pipeline import ROOT,OUT,save_json
from evaluation import event_counts,aggregate_events

rows=[r for r in json.loads((OUT/'manifest.json').read_text()) if r['split']=='validation']
data=[]
for r in rows:
    with np.load(ROOT/r['file']) as z:data.append({k:z[k] for k in ['t','labels','starts']})
offsets=np.r_[0,np.cumsum([r['windows'] for r in rows])]
results=[]
for folder in sorted((OUT/'experiments').glob('v2_*')):
    if not (folder/'result.json').exists():continue
    with np.load(folder/'validation_predictions.npz') as z:p=z['p']
    for high in [.5,.7,.8,.9,.95,.98,.995]:
        for alpha in [.3,.6]:
            for confirm in [1,3]:
                cfg=dict(high=high,low=max(.05,high-.1),alpha=alpha,confirm=confirm)
                events=[event_counts(z['t'],z['labels'],z['starts'],p[offsets[i]:offsets[i+1]],cfg) for i,z in enumerate(data)]
                results.append({'model':folder.name,'config':cfg,'validation':aggregate_events(events)})
eligible=[r for r in results if r['validation']['false_alarms_per_hour']<=2]
best=max(eligible,key=lambda r:r['validation']['sensitivity']) if eligible else None
save_json(ROOT/'reports/target_feasibility.json',{'split':'validation only','configurations':results,'best_sensitivity_under_two_false_events_per_hour':best,'target_met':any(r['validation']['sensitivity']>=.9 and r['validation']['median_delay_s'] is not None and r['validation']['median_delay_s']<=2 for r in eligible)})
print(json.dumps(best,indent=2))
