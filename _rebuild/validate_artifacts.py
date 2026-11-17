"""Validate actual processed shards and source integrity, not only synthetic fixtures."""
import json
import numpy as np
from data_pipeline import ROOT,OUT,fingerprint,save_json

def main():
    rows=json.loads((OUT/'manifest.json').read_text())
    assignments={};sources={};windows=0
    for r in rows:
        p=r['participant'];assert p not in assignments or assignments[p]==r['split']
        assignments[p]=r['split'];sources[r['source']]=r['sha256']
        with np.load(ROOT/r['file']) as archive:
            z={k:archive[k] for k in archive.files}
            assert z['x'].ndim==2 and z['x'].shape[1]==3
            assert np.isfinite(z['x']).all() and np.isin(z['labels'],[0,1]).all()
            assert np.allclose(np.diff(z['t']),.02,atol=1e-8)
            assert len(z['starts'])==r['windows']
            assert (z['starts']+100<=len(z['x'])).all()
            expected=np.array([z['labels'][s:s+100].mean() for s in z['starts']])
            np.testing.assert_allclose(z['overlap'],expected,atol=1e-7)
            windows+=len(z['starts'])
    for path,digest in sources.items():assert fingerprint(path)==digest,f'Source changed: {path}'
    report={'passed':True,'recording_segments':len(rows),'windows':windows,
            'participants':len(assignments),'verified_raw_source_checksums':len(sources),
            'checks':['finite 3-axis input','binary valid labels','50 Hz time grid','recording-bounded windows','exact overlap labels','disjoint participants','unchanged source SHA256']}
    save_json(ROOT/'reports/data_validation.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
