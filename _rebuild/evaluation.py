"""Window metrics and strictly causal event evaluation at decision times."""
import numpy as np
from sklearn.metrics import confusion_matrix,average_precision_score,roc_auc_score
from data_pipeline import runs

def metrics(y,p,threshold=.5):
    y=np.asarray(y); p=np.asarray(p)
    tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    return {'n':len(y),'accuracy':float((tp+tn)/max(len(y),1)),
        'precision':float(tp/max(tp+fp,1)),'recall':float(tp/max(tp+fn,1)),
        'specificity':float(tn/max(tn+fp,1)),'f1':float(2*tp/max(2*tp+fp+fn,1)),
        'pr_auc':float(average_precision_score(y,p)) if y.sum() else None,
        'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]]}

class Alarm:
    def __init__(self,high=.5,low=.3,confirm=2,alpha=.6):
        self.high=high;self.low=low;self.confirm=confirm;self.alpha=alpha
        self.reset()
    def reset(self): self.ema=0.;self.count=0;self.active=False
    def step(self,p,valid=True):
        if not valid or not np.isfinite(p): self.reset();return False
        self.ema=self.alpha*p+(1-self.alpha)*self.ema
        self.count=self.count+1 if self.ema>=self.high else 0
        if self.active and self.ema<self.low: self.active=False
        elif not self.active and self.count>=self.confirm: self.active=True
        return self.active

def event_counts(t,labels,starts,p,config):
    # Predictions apply from window END onward, never from window start.
    decision_times=t[starts+99]
    observed=(t>=decision_times[0])&(t<=decision_times[-1]+.2)
    truth=[(float(t[a]),float(t[b-1]+.02)) for a,b in runs((labels==1)&observed)]
    alarm=Alarm(**config)
    active=np.array([alarm.step(float(q)) for q in p])
    predicted=[(float(decision_times[a]),float(min(decision_times[b-1]+.2,t[-1]+.02))) for a,b in runs(active)]
    # Greedy chronological one-to-one nonzero-overlap matching; duplicate alerts
    # against an already matched episode count as false events.
    used=set();delays=[];matched=0
    for a,b in predicted:
        match=next((i for i,(c,d) in enumerate(truth) if i not in used and min(b,d)>max(a,c)),None)
        if match is not None:
            used.add(match);matched+=1;delays.append(max(0.,a-truth[match][0]))
    coverage=0.
    for c,d in truth:
        coverage+=sum(max(0.,min(b,d)-max(a,c)) for a,b in predicted)
    return {'events':len(truth),'predicted_events':len(predicted),'matched':matched,
            'false_events':len(predicted)-matched,'valid_seconds':float(observed.sum()/50),
            'fog_seconds':sum(d-c for c,d in truth),'covered_seconds':coverage,'delays':delays,
            'excluded_startup_seconds':float((~observed).sum()/50)}

def aggregate_events(rows):
    sums={k:sum(r[k] for r in rows) for k in ['events','predicted_events','matched','false_events','valid_seconds','fog_seconds','covered_seconds','excluded_startup_seconds']}
    delays=[v for r in rows for v in r['delays']]
    sums.update(sensitivity=sums['matched']/max(1,sums['events']),
                precision=sums['matched']/max(1,sums['predicted_events']),
                false_alarms_per_hour=sums['false_events']*3600/max(.001,sums['valid_seconds']),
                coverage=sums['covered_seconds']/max(.001,sums['fog_seconds']),
                missed_events=sums['events']-sums['matched'],
                median_delay_s=float(np.median(delays)) if delays else None,
                p90_delay_s=float(np.percentile(delays,90)) if delays else None)
    return sums
