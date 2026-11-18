import json
from pathlib import Path
import numpy as np
from src.data.daphnet_dataset import load_daphnet
from src.data.transforms import resample_causal,valid_segments
from src.features.windowing import window_indices
from src.evaluation.metrics import evaluate_traces
from src.inference.postprocessing import AlarmConfig
root=Path('reports')
files=sorted(root.glob('*/*validation_probabilities.npz'))
ref=json.loads((root/'feature_comparison_001/comparison.json').read_text())
records=[]; keep=[]; ys=[]
for rec in load_daphnet('daphnet'):
 if ref['splits'][rec.subject_key]!='validation': continue
 for seg in valid_segments(rec):
  if len(seg.timestamps)/seg.fs<20: continue
  r=resample_causal(seg); starts,y,_=window_indices(r,window_seconds=5)
  if len(y): records.append(r); keep.append(starts//10); ys.append(y)
y=np.concatenate(ys); streams=[]
for f in files:
 with np.load(f) as z:
  arr=[]
  for i,r in enumerate(records):
   np.testing.assert_allclose(z[f'times_{i}'],r.timestamps[499::10])
   arr.append(z[f'probabilities_{i}'])
  streams.append(arr)
scores=np.array([np.concatenate([p[k] for p,k in zip(a,keep)]) for a in streams])
best=(-1,-1,-1)
from itertools import combinations
for n in range(1,len(files)+1):
 for ids in combinations(range(len(files)),n):
  p=scores[list(ids)].mean(axis=0); order=np.argsort(-p); s=p[order]; lab=y[order]; idx=np.flatnonzero(np.r_[s[:-1]!=s[1:],True])
  tp=np.cumsum(lab)[idx]; fp=idx+1-tp; ba=.5*(1+tp/y.sum()-fp/(len(y)-y.sum()))
  for j in np.flatnonzero(ba==ba.max()):
   if ba[j]<best[0]: continue
   t=float(np.nextafter(s[idx[j]],-np.inf)); traces=[(r,r.timestamps[499::10],np.mean([streams[k][i] for k in ids],axis=0)) for i,r in enumerate(records)]
   metrics=evaluate_traces(traces,AlarmConfig(t)); m=metrics['pooled']; key=(ba[j],-m['fa_per_hour'],-fp[j])
   if key>best:
    best=key; result=dict(members=[str(files[k]) for k in ids],combination='equal mean of positive probabilities',threshold=t,balanced_accuracy=float(ba[j]),window_sensitivity=float(tp[j]/y.sum()),window_specificity=float(1-fp[j]/(len(y)-y.sum())),events=metrics)
print(json.dumps(result)); (root/'accuracy_first.json').write_text(json.dumps(result,indent=2))
