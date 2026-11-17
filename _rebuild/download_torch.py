"""Resumable ranged download of the official CUDA wheel; no environment mutation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
import time
URL='https://download-r2.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp314-cp314-win_amd64.whl'
ROOT=Path(__file__).resolve().parent/'artifacts'/'downloads'
ROOT.mkdir(exist_ok=True)
size=1998854405
chunk=32*1024*1024
def fetch(i):
    start=i*chunk;end=min(size,start+chunk)-1
    path=ROOT/f'{i}.part'
    for attempt in range(10):
        offset=path.stat().st_size if path.exists() else 0
        if offset==end-start+1:return path
        try:
            request=urllib.request.Request(URL,headers={'Range':f'bytes={start+offset}-{end}'})
            with urllib.request.urlopen(request,timeout=90) as response:
                if response.status!=206:raise RuntimeError('Server did not honor range')
                with path.open('ab') as f:
                    while block:=response.read(1024*1024):f.write(block)
            print('downloaded',i,flush=True)
        except Exception as e:
            print('retry',i,attempt,str(e),flush=True);time.sleep(2)
    raise RuntimeError(f'Failed part {i}')
with ThreadPoolExecutor(max_workers=6) as pool:
    paths=list(pool.map(fetch,range((size+chunk-1)//chunk)))
target=ROOT/'torch-2.14.1+cu130-cp314-cp314-win_amd64.whl'
with target.open('wb') as out:
    for path in paths:
        with path.open('rb') as inp:
            while block:=inp.read(1024*1024):out.write(block)
print('COMPLETE',target,flush=True)
