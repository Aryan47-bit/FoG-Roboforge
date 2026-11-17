"""Audited, recording-bounded accelerometer pipeline. Raw inputs are read-only."""
from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent
OUT = ROOT / 'artifacts'
VERSION = 'causal-50hz-v1'

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def recordings():
    for path in sorted((RAW/'daphnet').rglob('S*R*.txt')):
        a = pd.read_csv(path, sep=r'\s+', header=None).to_numpy()
        if a.shape[1] != 11 or not np.isin(a[:, 10], [0, 1, 2]).all():
            raise ValueError(f'Invalid Daphnet schema: {path}')
        yield dict(dataset='daphnet', participant='daphnet:'+path.stem[:3],
                   session=path.stem[3:], recording=path.stem, side='unknown',
                   location='shank', units='mg', source=str(path), sha256=fingerprint(path),
                   annotation='video annotation: 0 invalid, 1 normal, 2 FoG'), a[:,0]/1000, a[:,1:4]*0.00980665, np.where(a[:,10]==0,-1,a[:,10]-1)
    # On-disk grouping handles interleaved rows without holding all recordings in RAM.
    path = RAW/'fogstar'/'sensor_data.csv'
    digest = fingerprint(path)
    db = OUT/'fogstar_index.sqlite'
    marker = OUT/'fogstar_index.sha256'
    conn = sqlite3.connect(db)
    if not marker.exists() or marker.read_text() != digest:
        conn.execute('DROP TABLE IF EXISTS samples')
        columns = ['timestamp','subjectID','sessionID','taskID','fog'] + [f'ankle{s}_acc_{c}' for s in 'LR' for c in 'xyz']
        for chunk in pd.read_csv(path, usecols=columns, chunksize=50000):
            if not chunk.fog.isin([0,1]).all():
                raise ValueError('FoG-STAR has missing/unknown labels')
            chunk.to_sql('samples', conn, if_exists='append', index=False)
        conn.execute('CREATE INDEX recording_index ON samples(subjectID,sessionID,taskID,timestamp)')
        conn.commit()
        marker.write_text(digest)
    keys = conn.execute('SELECT DISTINCT subjectID,sessionID,taskID FROM samples ORDER BY 1,2,3').fetchall()
    for subject, session, task in keys:
        f = pd.read_sql_query('SELECT * FROM samples WHERE subjectID=? AND sessionID=? AND taskID=? ORDER BY timestamp',conn,params=(subject,session,task))
        t = f.timestamp.to_numpy(float)
        # Official description calls these ms; this local file increments by 1/60.
        # Accept only either documented 60 Hz encoding; never silently infer arbitrary rates.
        dt = np.median(np.diff(t))
        if abs(dt-1000/60)<0.01: t=t/1000
        elif abs(dt-1/60)>1e-5: raise ValueError(f'Unknown timestamp scale {dt}')
        for side in 'LR':
            yield dict(dataset='fogstar',participant=f'fogstar:{subject}',session=str(session),
                       recording=f'{subject}_{session}_{task}_{side}',side=side,location='ankle',
                       units='g',source=str(path),sha256=digest,
                       annotation='fog: 0 no FoG, 1 FoG; official Zenodo v3'), t, f[[f'ankle{side}_acc_{c}' for c in 'xyz']].to_numpy(float)*9.80665, f.fog.to_numpy(int)
    conn.close()

def audit_record(meta,t,x,y):
    dt=np.diff(t)
    if not len(dt) or np.any(dt<=0): raise ValueError(f'Nonmonotonic recording: {meta}')
    fs=(len(t)-1)/(t[-1]-t[0])
    expected=64 if meta['dataset']=='daphnet' else 60
    if abs(fs-expected)>0.1: raise ValueError(f'Unexpected rate {fs}')
    finite=np.isfinite(x).all(axis=1)
    return {**meta,'processing_version':VERSION,'samples':len(t),'sampling_rate_hz':float(fs),
            'nominal_rate_hz':expected,'duplicate_timestamps':int((dt==0).sum()),
            'max_interval_s':float(dt.max()),'missing_accel_rows':int((~finite).sum()),
            'outside_experiment_samples':int((y<0).sum()),
            'fog_seconds':float((y==1).sum()/expected),'normal_seconds':float((y==0).sum()/expected),
            'episodes':len(runs(y==1)),'outside_device_2g_rows':int((np.abs(x)>19.6133).any(axis=1).sum()),
            'orientation':'native axes; no unsupported anatomical transform'}

def preprocess(t,x,y,fs):
    """Causal source anti-aliasing; split at ANY missing/invalid sample or timing gap.

    Previous-sample hold uses no future sample. Source filter delay is retained in
    real timestamps, never removed by shifting ground truth. One second warm-up
    plus the two-second window creates explicitly excluded observation periods.
    """
    good=np.isfinite(x).all(axis=1)&(y>=0)
    cuts=np.r_[True,np.diff(t)>1.6/fs]
    for start,end in runs(good):
        boundaries=np.r_[start,np.flatnonzero(cuts[start+1:end])+start+1,end]
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            if b-a < 3*fs: continue
            filtered=sosfilt(butter(4,20,fs=fs,output='sos'),x[a:b],axis=0)
            grid=np.arange(t[a]+1,t[b-1],1/50)
            idx=np.searchsorted(t[a:b],grid,side='right')-1
            yield grid,np.clip(filtered[idx],-19.6133,19.6133).astype('float32'),y[a:b][idx].astype('int8')

def split_participants(manifest):
    result={}
    rng=np.random.default_rng(20261117)
    for dataset in sorted({m['dataset'] for m in manifest}):
        people=sorted({m['participant'] for m in manifest if m['dataset']==dataset})
        rng.shuffle(people)
        n=max(2,round(len(people)*.2))
        for split, ids in [('test',people[:n]),('validation',people[n:2*n]),('train',people[2*n:])]:
            result.update({p:split for p in ids})
    return result

def build():
    OUT.mkdir(exist_ok=True)
    manifest=[]
    # Complete read-only audit before processing or training.
    for meta,t,x,y in recordings():
        manifest.append(audit_record(meta,t,x,y))
    save_json(ROOT/'reports'/'dataset_audit.json',manifest)
    split_path=OUT/'splits.json'
    splits=split_participants(manifest)
    if split_path.exists() and json.loads(split_path.read_text())!=splits:
        raise RuntimeError('Existing frozen splits differ; use a new versioned output directory')
    save_json(split_path,splits)
    index=[]
    cache=OUT/'recordings'; cache.mkdir(exist_ok=True)
    for meta,t,x,y in recordings():
        fs=64 if meta['dataset']=='daphnet' else 60
        for segment,(grid,acc,labels) in enumerate(preprocess(t,x,y,fs)):
            starts=np.arange(0,len(grid)-99,10,dtype=np.int64)
            if not len(starts): continue
            overlap=np.array([labels[s:s+100].mean() for s in starts],dtype='float32')
            name=f"{meta['dataset']}_{meta['recording']}_{segment}"
            file=cache/(name+'.npz')
            np.savez_compressed(file,x=acc,t=grid,labels=labels,starts=starts,overlap=overlap)
            index.append({**meta,'segment':segment,'file':str(file.relative_to(ROOT)),
                          'split':splits[meta['participant']],'windows':len(starts),
                          'sampling_rate_hz':fs,'processing_version':VERSION,
                          'valid_seconds':len(grid)/50})
        print('processed',meta['dataset'],meta['recording'],flush=True)
    save_json(OUT/'manifest.json',index)
    frame=pd.DataFrame(manifest)
    frame.groupby(['dataset','participant']).agg(samples=('samples','sum'),fog_seconds=('fog_seconds','sum'),normal_seconds=('normal_seconds','sum'),missing_rows=('missing_accel_rows','sum')).to_csv(ROOT/'reports'/'participant_statistics.csv')
    save_json(ROOT/'reports'/'availability.json',{
        'daphnet':'17 local recordings; actual rate ~64 Hz',
        'fogstar':'local bilateral labelled CSV; official units g',
        'multimodal':'Only unaligned repository samples, no verified participant/session identity. Excluded from supervised experiments; full labelled release required.',
        'multimodal_files':[str(p.relative_to(RAW)) for p in (RAW/'multimodal_parkinson').rglob('*') if p.is_file() and p.suffix in ('.txt','.csv')],
        'fogstar_timestamp_caveat':'Official README says milliseconds, local file step is 0.0166667; treated as seconds at documented 60 Hz.',
        'source_urls':['https://zenodo.org/records/17838806','https://data.mendeley.com/datasets/r8gmbtv7w2/3'],
        'bilateral_duration_caveat':'FoG-STAR per-side durations count the same observation twice; report sides separately.',
    })
    print('AUDIT AND PROCESSING COMPLETE',len(index),'segments',sum(m['windows'] for m in index),'windows')

if __name__=='__main__': build()
