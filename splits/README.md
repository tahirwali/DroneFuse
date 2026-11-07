# Splits

Made once (2026-11-05) from the official bag indexes and Leica ground-truth bags by
`dronefuse.windows.build_splits`, and loaded by every later step. **Never edit by hand.**

- `windows.csv`: 7,445 analysis windows (1.0 s long, every 0.5 s), one row each: recording, version,
  drone type, start/end bag time, time block (0–9) and split. Windows exist only where every sensor is
  recording **and** the Leica tracker has the drone (at least one ground-truth message inside the window).
- `splits_info.json`: the parameters, each recording's usable span, and window counts per protocol.

**Time blocks.** Each recording's usable span is cut into 10 equal blocks assigned `T T V T S T T V T S`
(train / val / test = 60 / 20 / 20 %), so every split samples the start, middle and end of the flight.
A window must lie inside one block and at least 1 s from a neighbouring block of another split, so windows
of different splits are always at least 2 s apart.

**Protocols** (`dronefuse.windows.protocol_sets`):

| Protocol | Train / val from | Test | Unknown types at test | Train | Val | Test known | Test unknown |
|---|---|---|---|---:|---:|---:|---:|
| `closed` | all 12 recordings | test blocks | – | 4493 | 1469 | 1483 | 0 |
| `v1_to_v2` | V1 (rooftop) | **all** V2/V3 windows | M30 | 1306 | 428 | 4056 | 1226 |
| `v2_to_v1` | V2/V3 (carpark) | **all** V1 windows | Avata, Mavic2 | 3187 | 1041 | 1108 | 1055 |
| `loo_Avata` | 5 other types | test blocks | Avata | 4120 | 1349 | 1361 | 122 |
| `loo_M30` | 5 other types | test blocks | M30 | 3754 | 1228 | 1237 | 246 |
| `loo_M300` | 5 other types | test blocks | M300 | 3639 | 1186 | 1200 | 283 |
| `loo_Mavic2` | 5 other types | test blocks | Mavic2 | 4226 | 1384 | 1395 | 88 |
| `loo_Mavic3` | 5 other types | test blocks | Mavic3 | 3333 | 1089 | 1098 | 385 |
| `loo_Phantom4` | 5 other types | test blocks | Phantom4 | 3393 | 1109 | 1124 | 359 |
| `xrec_s1_to_s2` | car park, flight 1 of M300, M30, Phantom 4 | **all** windows of flight 2, + `m3e` test blocks | Mavic3 | 1096 | 358 | 1861 | 321 |
| `xrec_s2_to_s1` | car park, flight 2 of M300, M30, Phantom 4 | **all** windows of flight 1, + `m3e` test blocks | Mavic3 | 1126 | 365 | 1817 | 321 |

Class = drone type; Mavic 3 (V1) and Mavic 3 Enterprise (V2/V3 `m3e`) are one class, `Mavic3`.
Validation never contains an unknown type, so rejection thresholds are set without seeing unknowns.

The cross-recording protocols (`xrec_*`, added 2026-11-07) test on flights never seen in training, for known and
unknown types alike, so a model cannot score by recognising the recording (in `closed` and `loo_*`, training
and test windows of a known type come from the same flight).
