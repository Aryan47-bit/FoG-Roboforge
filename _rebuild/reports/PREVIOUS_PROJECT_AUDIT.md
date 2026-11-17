# Previous project and hardware audit

Source inspected: `Old-project/Parkinson's FoG`, especially
`firmware/ESP32_FoG/model_config.h`, `imu_sensor.cpp`, `src/preprocess.py` and
`reports/pipeline_eval/pipeline_eval_report.md`. Old files are preserved.

Pins in source: MPU6050 SDA 21, SCL 22, motor 13, BOOT calibration button 0,
status LED 2. I2C address 0x68, 400 kHz. Accelerometer ±2 g /16384 LSB/g;
registers PWR_MGMT_1=0x01, CONFIG=0x03, SMPLRT_DIV=0x13, ACCEL_CONFIG=0x00.
Motor LEDC channel 0, 1 kHz, 8-bit, duty 220. The documented driver is a low-side
transistor/MOSFET with gate pull-down; physical components were not inspected.
The rebuilt integration preserves these assignments and sensor registers.

The old model consumed six channels including gyroscope, unlike the new required
three-accelerometer input. It used left ankle only, 100-sample windows and 25-sample
stride, and mixed pre-FoG/transition labels with active freezing. The firmware had
EMA smoothing, consecutive confirmations, hysteresis and refractory timing.
Those target and decision differences prevent treating headline classifier scores
as real alarm performance.

The saved pipeline report states default float test false alerts/hour 92.81,
strict sensitivity 0/18 and lenient sensitivity 12/18. These are historical reported
numbers, not rerun results. Definitions and splits differ from this rebuild, so
no direct improvement claim is valid. Old latency and resource reports do not
substitute for physical measurements on the new artifact.

Previous rebuild reports in the repository reference another Windows user and
Intel Arc `tf_arc`. This machine instead has Python 3.14, TensorFlow 2.22.0rc0
without GPU support, and an RTX 4050 with driver 592.82. The user explicitly
requested NVIDIA training. A project-local CUDA PyTorch environment is used.
The earlier Daphnet 62.5 Hz calculation was incorrect; actual timestamps span
approximately 64 samples/second. Historical caches/weights were not reused.

Reusable: hardware assignments, documented electrical interface and sensor
register setup. Replaced: data loaders, splitting, labels, preprocessing, models,
training, event evaluation, export, and firmware task/decision implementation.

The staged work is audit → source harmonization/tests → individual baselines →
combined/holdout/side/representation experiments → validation-only selection →
INT8 conversion/parity → board validation. The user's direct build-and-train
request authorizes software stages; phase-stop prompts embedded in plan.txt are
not treated as additional user approval requests. Missing data and unavailable
physical hardware are reported as limitations rather than fabricated completion.
