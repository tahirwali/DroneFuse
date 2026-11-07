# T3 - The recording check on TSMS-Drone and DroneRFc-MM

**Notebook:** `notebooks/t3_extra_datasets.ipynb` | **Code:** `dronefuse/extra.py` | **Tests:** `tests/test_extra.py`
**Kaggle:** GPU T4, Internet On. Inputs: `dronefuse-code`, `dronefuse-tsms` (T1, all parts), `dronefuse-rfcmm-core` (T2).

The same three questions as on MMAUD, on two independent datasets:
1. Probes (train -> val, balanced accuracy): drone type, and session (TSMS: the 15 distances of one target;
   DroneRFc-MM: flight 1 vs flight 2 of A1, E1, G1).
2. Accuracy within sessions (ten time blocks per session, MMAUD pattern) vs across sessions (TSMS: train on
   2, 6, ..., 30 m, test on 4, 8, ..., 28 m and the reverse; DroneRFc-MM: train on one flight of A1, E1, G1,
   test on the other, with the test blocks of B1, C1, F1 as unknown types).
3. Unknown-type rejection (MSP AUROC) with one type left out: TSMS within-session vs cross-distance;
   DroneRFc-MM within-session.

Features: ConvNeXt-T (ImageNet) for TSMS images and the PTZ/fisheye middle frame of each 1 s window;
log-mel mean and spread per microphone channel (4 x 128); 34 statistics of the radar returns. Model: standardise,
PCA to 128 when larger, logistic regression. Fusion: concatenation and late fusion (average probabilities).
95 % intervals: block bootstrap over test time blocks (1000).

DroneRFc-MM windows: 1 s over the time both cameras recorded (cameras, audio and radar cover only the middle part
of each flight log; 2,284 windows in 9 sessions). Drone type = session name without "-2".
TSMS rows: measurements present for all three sensors with the same (target, distance, index).

Checked: 4 offline tests; parsers on real G1 files (audio 179/179 windows, radar 179/179, PTZ 179/179);
a dry run of the notebook on synthetic TSMS tars.
