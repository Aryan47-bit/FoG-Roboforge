"""GPU-only training; frozen participant splits; validation-only selection."""
import argparse
import json
import random
import time
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from data_pipeline import ROOT,OUT,save_json,fingerprint
from models import Detector,verify_gpu
from evaluation import metrics,event_counts,aggregate_events

class Windows(Dataset):
    def __init__(self,records,mean=None,std=None,overlap=.5,representation='gravity'):
        self.records=records;self.cache=OrderedDict();self.mean=mean;self.std=std
        self.overlap=overlap;self.representation=representation
        self.offsets=np.r_[0,np.cumsum([r['windows'] for r in records])]
    def __len__(self):return int(self.offsets[-1])
    def recording(self,i):
        if i not in self.cache:
            with np.load(ROOT/self.records[i]['file']) as z: self.cache[i]={k:z[k] for k in z.files}
            if self.representation=='dynamic':
                from scipy.signal import lfilter
                x=self.cache[i]['x'];self.cache[i]['x']=x-lfilter([.02],[1,-.98],x,axis=0)
            if len(self.cache)>256:self.cache.popitem(last=False)
        return self.cache[i]
    def __getitem__(self,index):
        i=int(np.searchsorted(self.offsets,index,side='right')-1);j=int(index-self.offsets[i])
        z=self.recording(i);s=z['starts'][j];x=z['x'][s:s+100]
        if self.mean is not None:x=(x-self.mean)/self.std
        return np.asarray(x,dtype='float32'),np.float32(z['overlap'][j]>=self.overlap)

def normalization(records,representation):
    ds=Windows(records,representation=representation);s=np.zeros(3);ss=np.zeros(3);n=0
    for i in range(len(records)):
        x=ds.recording(i)['x'].astype('float64');s+=x.sum(0);ss+=(x*x).sum(0);n+=len(x)
    mean=s/n;std=np.sqrt(np.maximum(ss/n-mean*mean,1e-6))
    return mean.astype('float32'),std.astype('float32')

def predict(model,ds):
    model.eval();out=[]
    with torch.no_grad():
        for x,_ in DataLoader(ds,batch_size=512,shuffle=False):
            out.append(torch.sigmoid(model(x.cuda())).cpu().numpy())
    return np.concatenate(out)

def evaluate(ds,p,config):
    y=np.concatenate([(ds.recording(i)['overlap']>=ds.overlap).astype(int) for i in range(len(ds.records))])
    ev=[];groups={}
    for i,r in enumerate(ds.records):
        z=ds.recording(i);a,b=ds.offsets[i:i+2]
        counts=event_counts(z['t'],z['labels'],z['starts'],p[a:b],config);ev.append(counts)
        for category,key in [('dataset',r['dataset']),('participant',r['participant']),('side',r['side'])]:
            entry=groups.setdefault(category+':'+key,{'y':[],'p':[],'events':[]})
            entry['y'].extend(y[a:b]);entry['p'].extend(p[a:b]);entry['events'].append(counts)
    return {'window':metrics(y,p,config['high']),'event':aggregate_events(ev),
            'groups':{k:{'window':metrics(v['y'],v['p'],config['high']),'event':aggregate_events(v['events'])} for k,v in groups.items()}}

def train_one(name,records,kind,epochs,representation='gravity',overlap=.5,sampling='balanced'):
    out=OUT/'experiments'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
    seed=20261117;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    parts={s:[r for r in records if r['split']==s] for s in ['train','validation','test']}
    sets={s:{r['participant'] for r in rows} for s,rows in parts.items()}
    assert all(not sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    if any(not rows for rows in parts.values()):raise ValueError('Empty split')
    mean,std=normalization(parts['train'],representation)
    ds={s:Windows(rows,mean,std,overlap,representation) for s,rows in parts.items()}
    counts={p:sum(r['windows'] for r in parts['train'] if r['participant']==p) for p in sets['train']}
    dataset_people={d:len({r['participant'] for r in parts['train'] if r['dataset']==d}) for d in {r['dataset'] for r in parts['train']}}
    weights=np.concatenate([np.full(r['windows'],1/(counts[r['participant']]*dataset_people[r['dataset']])) for r in parts['train']])
    if sampling=='uniform':weights=np.ones_like(weights)
    # Epoch samples are bounded; participant- and dataset-balanced replacement is recorded.
    sampler=WeightedRandomSampler(torch.from_numpy(weights),min(len(ds['train']),24000),replacement=True,generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(ds['train'],batch_size=256,sampler=sampler,num_workers=0)
    model=Detector(kind).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=2)
    ytrain=np.concatenate([ds['train'].recording(i)['overlap']>=overlap for i in range(len(parts['train']))])
    # Weighted prevalence must match the sampling distribution.
    prevalence=float(np.average(ytrain,weights=weights));pos_weight=min(10.,(1-prevalence)/max(prevalence,.001))
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device='cuda'))
    yval=np.concatenate([ds['validation'].recording(i)['overlap']>=overlap for i in range(len(parts['validation']))])
    best=-1;stale=0;history=[];start_epoch=0
    if (out/'last.pt').exists():
        resume=torch.load(out/'last.pt',weights_only=False)
        model.load_state_dict(resume['state_dict']);opt.load_state_dict(resume['optimizer'])
        if 'resume' in resume:
            state=resume['resume'];best=state['best'];stale=state['stale'];history=state['history']
            scheduler.load_state_dict(state['scheduler'])
            sampler.generator.set_state(state['sampler_rng'].cpu())
            torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
            start_epoch=resume['epoch']
    for epoch in range(start_epoch,epochs):
        model.train();loss_sum=0.;n=0;begin=time.perf_counter()
        for x,y in loader:
            opt.zero_grad();loss=lossfn(model(x.cuda()),y.cuda())
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            loss_sum+=loss.item()*len(x);n+=len(x)
        p=predict(model,ds['validation']);score=metrics(yval,p)['pr_auc'] or 0.
        row={'epoch':epoch+1,'loss':loss_sum/n,'validation_pr_auc':score,'seconds':time.perf_counter()-begin}
        history.append(row);print(name,row,flush=True)
        scheduler.step(score)
        checkpoint={'state_dict':model.state_dict(),'kind':kind,'mean':mean.tolist(),'std':std.tolist(),
                    'representation':representation,'overlap':overlap,'epoch':epoch+1}
        if score>best+.0001:best=score;stale=0;torch.save(checkpoint,out/'best.pt')
        else:stale+=1
        torch.save({**checkpoint,'optimizer':opt.state_dict(),'resume':{
            'best':best,'stale':stale,'history':history,'scheduler':scheduler.state_dict(),
            'sampler_rng':sampler.generator.get_state(),'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state()}},out/'last.pt')
        if stale>=5:break
    checkpoint=torch.load(out/'best.pt',weights_only=False);model.load_state_dict(checkpoint['state_dict'])
    p=predict(model,ds['validation']);best_score=-float('inf');selected=None
    # Thresholds chosen using validation events only. Predeclared tradeoff: event F1
    # minus 0.002 * false alarms/hour. This is an engineering selection rule, not a clinical target.
    for threshold in [.25,.35,.45,.55,.65,.75]:
        for confirm in [1,2,3]:
            cfg={'high':threshold,'low':max(.05,threshold-.2),'confirm':confirm,'alpha':.6}
            result=evaluate(ds['validation'],p,cfg);e=result['event']
            f1=2*e['precision']*e['sensitivity']/max(1e-9,e['precision']+e['sensitivity'])
            score=f1-.002*e['false_alarms_per_hour']
            if score>best_score:best_score=score;selected=cfg;val=result
    save_json(out/'history.json',history)
    np.savez_compressed(out/'validation_predictions.npz',p=p,y=yval)
    result={'name':name,'kind':kind,'parameters':sum(p.numel() for p in model.parameters()),
            'source_sha256':{f:fingerprint(ROOT/f) for f in ['data_pipeline.py','models.py','train.py','evaluation.py']},
            'dataset_sha256':sorted({r['sha256'] for r in records}),
            'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(),
            'device':torch.cuda.get_device_name(0),'seed':seed,'epochs':len(history),
            'representation':representation,'overlap':overlap,'normalization':{'mean':mean.tolist(),'std':std.tolist()},
            'sampling':sampling+' with replacement, maximum 24000 windows/epoch',
            'positive_loss_weight':pos_weight,'validation_selection_score':best_score,'alarm':selected,
            'splits':{s:sorted(ps) for s,ps in sets.items()},'validation':val}
    # Save selection before ever inspecting held-out test.
    save_json(out/'selection.json',result)
    test_p=predict(model,ds['test']);result['test']=evaluate(ds['test'],test_p,selected)
    people=[v['event'] for k,v in result['test']['groups'].items() if k.startswith('participant:')]
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(1000):
        sampled=[people[i] for i in rng.integers(len(people),size=len(people))]
        events=sum(v['events'] for v in sampled);seconds=sum(v['valid_seconds'] for v in sampled)
        if events:boot.append([sum(v['matched'] for v in sampled)/events,sum(v['false_events'] for v in sampled)*3600/seconds])
    result['test']['participant_bootstrap_95_percent_interval']={
        'sensitivity':np.percentile(boot,[2.5,97.5],axis=0)[:,0].tolist() if boot else None,
        'false_alarms_per_hour':np.percentile(boot,[2.5,97.5],axis=0)[:,1].tolist() if boot else None,
        'method':'1000 participant resamples; both sides kept together; resamples without positive events omitted',
        'participants':len(people)}
    np.savez_compressed(out/'test_predictions.npz',p=test_p)
    save_json(out/'result.json',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--suite',choices=['baseline','extended'],default='baseline');args=parser.parse_args()
    torch.set_num_threads(4)
    save_json(ROOT/'reports'/'gpu_verification.json',verify_gpu())
    records=json.loads((OUT/'manifest.json').read_text());results=[]
    for dataset in ['daphnet','fogstar','combined']:
        rows=[r for r in records if dataset=='combined' or r['dataset']==dataset]
        for kind in ['cnn','rescnn','tcn']:
            results.append(train_one(dataset+'_'+kind,rows,kind,args.epochs))
    if args.suite=='extended':
        for held in ['daphnet','fogstar']:
            rows=[dict(r,split='test') if r['dataset']==held else r for r in records if r['dataset']==held or r['split']!='test']
            results.append(train_one('heldout_'+held,rows,'cnn',args.epochs))
        for side in ['L','R']:
            rows=[r for r in records if r['dataset']=='fogstar' and (r['split']=='test' or r['side']==side)]
            results.append(train_one('train_side_'+side,rows,'cnn',args.epochs))
        for representation,overlap in [('dynamic',.5),('gravity',.2),('gravity',.8)]:
            results.append(train_one(f'combined_cnn_{representation}_{overlap}',records,'cnn',args.epochs,representation,overlap))
        results.append(train_one('combined_cnn_uniform',records,'cnn',args.epochs,sampling='uniform'))
        # Grouped development CV never includes any frozen final test participant.
        dev=[r for r in records if r['split']!='test'];folds={}
        rng=np.random.default_rng(20261117)
        for dataset in ['daphnet','fogstar']:
            people=sorted({r['participant'] for r in dev if r['dataset']==dataset});rng.shuffle(people)
            for fold,ids in enumerate(np.array_split(people,3)):
                folds.update({str(p):fold for p in ids})
        for fold in range(3):
            remaining=sorted(p for p,f in folds.items() if f!=fold)
            val=set()
            for dataset in ['daphnet','fogstar']:
                people=[p for p in remaining if p.startswith(dataset+':')]
                val.update(people[:max(1,len(people)//5)])
            rows=[dict(r,split='test' if folds[r['participant']]==fold else ('validation' if r['participant'] in val else 'train')) for r in dev]
            results.append(train_one(f'development_cv_{fold}',rows,'cnn',args.epochs))
    # Deployment candidate is selected only among combined-data models by validation.
    candidates=[r for r in results if r['name'].startswith('combined_')]
    best=max(candidates,key=lambda r:r['validation_selection_score'])
    save_json(ROOT/'reports'/'experiment_summary.json',results)
    save_json(OUT/'selected.json',{'experiment':best['name'],'criterion':'maximum validation event F1 - 0.002 * false alarms/hour','test_used_for_selection':False})
    print('SELECTED',best['name'],flush=True)

if __name__=='__main__':main()
