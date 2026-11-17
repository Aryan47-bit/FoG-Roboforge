"""Collect serial S lines into a pseudonymous session CSV and metadata."""
import argparse,csv,json,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',required=True);p.add_argument('--participant',required=True)
    p.add_argument('--side',choices=['L','R'],required=True);p.add_argument('--mounting',required=True)
    p.add_argument('--notes',default='');p.add_argument('--seconds',type=float,default=60);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    import serial
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'metadata.json').write_text(json.dumps({'participant':a.participant,'side':a.side,'mounting':a.mounting,'notes':a.notes,'units':'m/s^2','nominal_rate_hz':50,'annotation':'unlabelled; excluded from supervised training'},indent=2))
    with serial.Serial(a.port,115200,timeout=1) as port,open(a.out/'samples.csv','w',newline='') as f:
        port.write(b'v');writer=csv.writer(f);writer.writerow(['timestamp_ms','acc_x','acc_y','acc_z','status'])
        begin=time.monotonic();last=None
        try:
            while time.monotonic()-begin<a.seconds:
                line=port.readline().decode(errors='replace').strip().split(',')
                if len(line)!=6 or line[0]!='S':continue
                values=[float(v) for v in line[1:]]
                if last is not None and values[0]<=last:raise ValueError('Nonmonotonic device timestamp')
                last=values[0];writer.writerow(values)
        finally:port.write(b'v')

if __name__=='__main__':main()
