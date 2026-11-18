from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('model',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
data=args.model.read_bytes()
if data[4:8]!=b'TFL3':
    raise ValueError('Expected TFLite flatbuffer')
lines=[','.join(f'0x{b:02x}' for b in data[i:i+16]) for i in range(0,len(data),16)]
args.output.write_text('#pragma once\nalignas(16) const unsigned char fog_model[]={\n'+',\n'.join(lines)+'\n};\n')
