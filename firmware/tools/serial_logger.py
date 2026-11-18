import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import serial

parser = argparse.ArgumentParser(description="Capture ESP32 FoG telemetry")
parser.add_argument("port")
parser.add_argument("--seconds", type=float, default=600)
args = parser.parse_args()
path = Path(datetime.now().strftime("fog_%Y%m%d_%H%M%S.csv"))
previous = None
with serial.Serial(args.port, 115200, timeout=1) as device, path.open("x", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["host_time", "packet", "device_ms", "payload..."])
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        fields = device.readline().decode("ascii", errors="replace").strip().split(",")
        if len(fields) != {"S": 6, "P": 6, "E": 4, "H": 6}.get(fields[0], -1):
            continue
        writer.writerow([time.time(), *fields])
        stamp = int(fields[1])
        if fields[0] == "S":
            jitter = None if previous is None else ((stamp-previous) % 2**32)-20
            previous = stamp
        elif fields[0] == "P":
            print(f"{'#'*int(float(fields[2])*30):30} state={fields[3]} inference_us={fields[4]} last_sample_jitter_ms={jitter if previous is not None else None}")
print(path)
