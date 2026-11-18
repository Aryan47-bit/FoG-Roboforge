from pathlib import Path
import json
import numpy as np
from scipy.optimize import differential_evolution
from src.data.daphnet_dataset import load_daphnet
from src.data.transforms import resample_causal,valid_segments
from src.features.windowing import window_indices
from src.evaluation.metrics import evaluate_traces
from src.inference.postprocessing import AlarmConfig

def main():
    root=Path("reports")
    files=sorted(root.glob("*/*validation_probabilities.npz"))
    old=json.loads((root/"accuracy_first.json").read_text())
    splits=json.loads((root/"feature_comparison_001/comparison.json").read_text())["splits"]
    records=[]; indices=[]; labels=[]
    for rec in load_daphnet("daphnet"):
        if splits[rec.subject_key]!="validation": continue
        for seg in valid_segments(rec):
            if len(seg.timestamps)/seg.fs<20: continue
            r=resample_causal(seg)
            starts,y,_=window_indices(r,window_seconds=5)
            if len(y):
                records.append(r); indices.append(starts//10); labels.extend(y)
    streams=[]
    for path in files:
        with np.load(path) as z:
            for i,r in enumerate(records):
                np.testing.assert_allclose(z[f"times_{i}"],r.timestamps[499::10])
            streams.append([z[f"probabilities_{i}"] for i in range(len(records))])
    x=np.array([np.concatenate([p[k] for p,k in zip(a,indices)]) for a in streams])
    positive=np.array(labels)==1
    def accuracy(weights,threshold):
        predicted=weights@x>threshold
        return (predicted[positive].mean()+(~predicted[~positive]).mean())/2
    def objective(params):
        w=params[:-1]
        return -accuracy(w/max(w.sum(),1e-12),params[-1])
    seed=np.array([str(f) in old["members"] for f in files],float)
    seed/=seed.sum()
    candidates=[seed]
    for rng in (42,73):
        fit=differential_evolution(objective,[(0,1)]*(len(files)+1),x0=np.r_[seed,old["threshold"]],seed=rng,popsize=10,maxiter=100,tol=0.0001,polish=False)
        candidates.extend(row[:-1]/row[:-1].sum() for row in fit.population)
    best=(-1,-float("inf"))
    for w in candidates:
        scores=w@x
        order=np.argsort(-scores); ranked=scores[order]
        ends=np.flatnonzero(np.r_[ranked[:-1]!=ranked[1:],True])
        tp=np.cumsum(positive[order])[ends]; fp=ends+1-tp
        ba=(1+tp/positive.sum()-fp/(~positive).sum())/2
        if ba.max()<best[0]: continue
        for j in np.flatnonzero(ba==ba.max()):
            threshold=float(np.nextafter(ranked[ends[j]],-np.inf))
            traces=[(r,r.timestamps[499::10],w@np.stack([a[i] for a in streams])) for i,r in enumerate(records)]
            metrics=evaluate_traces(traces,AlarmConfig(threshold))
            key=(float(ba[j]),-metrics["pooled"]["fa_per_hour"])
            if key>best:
                best=key
                result=dict(weights={str(f):float(v) for f,v in zip(files,w)},threshold=threshold,balanced_accuracy=key[0],window_sensitivity=float(tp[j]/positive.sum()),window_specificity=float(1-fp[j]/(~positive).sum()),events=metrics,test_evaluated=False)
    (root/"weighted_accuracy.json").write_text(json.dumps(result,indent=2))
    print(result)
if __name__=="__main__":
    main()
