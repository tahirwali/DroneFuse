# Step 3 — Fusion with missing sensors and unknown-drone rejection

## What
One GPU notebook (`notebooks/step3_fusion.ipynb`) reads the Step 2 dataset `dronefuse-step2-features` and runs
11 methods × 9 protocols × 3 seeds. Every run is saved as soon as it ends (`runs/`, logits included), so new
analyses need no retraining and an interrupted run can be continued.

| Method | Idea |
|---|---|
| Camera / Audio / Avia / Mid-360 / Radar only | one sensor group; "no opinion" (counted wrong) where it is missing |
| Late fusion | mean probability of the single-group models that are present |
| Feature concatenation | all features, zeros where missing |
| LMMUAVC-style | learned missing slot per sensor, sensors dropped i.i.d. (p = 0.3) in training |
| **DroneFuse** | per-sensor encoders → attention that skips missing sensors; training removes sensors in the co-missing patterns of the training windows ("pattern transplant", p = 0.5) |
| DroneFuse (i.i.d.) / (mixed) | same network, i.i.d. dropping / pattern + light i.i.d. (ablations) |

Rejection of unknown types: max softmax, energy, and **availability-normalised energy** (energy standardised
by validation windows with the same number of sensor groups present). Threshold = keep 95 % of validation.

Outputs: `step3_report.txt`, `results.json`, `tables/*.tex` (main, rejection scores, accuracy by groups
present, test-time removal), `figures/*.png` (subset grids, accuracy vs. groups present).

## Why
- **Natural missingness is structured.** Sensors fail together (field of view, range, site); radar is
  practically absent at the V2/V3 carpark. LMMUAVC drops sensors independently. Training with observed
  patterns, and evaluating with both natural and forced removal, tests whether that matters.
- **Rejection should depend on what was observed.** A window seen by one sensor group is less certain than
  one seen by five; normalising the energy per number of groups present makes one threshold fair to both.
- **Same features, optimiser, epochs and early stopping for every method**, so differences come from the
  fusion and the dropping rule only.

## How (`dronefuse/fusion.py`, `dronefuse/step3.py`)
- Tokens: camera ConvNeXt avg+max (1536); Avia all-point descriptor (39); Mid-360 and radar foreground
  descriptors (39, 49); per audio node AST (768) + log-mel mean/std (128); each + log(1+count) scaled to [0,1].
  Standardisation statistics from present **training** windows only.
- Presence: camera/audio ≥ 1 frame; LiDAR/radar ≥ t points, t ∈ {1, 5, 25} chosen per protocol by DroneFuse
  validation accuracy (first seed) and then used by all methods.
- Encoders: LayerNorm → 256 → GELU → 128 (audio nodes share one encoder; sensor embeddings added).
  DroneFuse: CLS token + 2 pre-norm transformer layers (4 heads), key-padding mask for missing sensors.
- Training: AdamW (lr 1e-3, wd 1e-2), one-cycle schedule, 40 epochs, batch 256, class-weighted cross-entropy,
  best epoch by validation accuracy.
- Test-time removal (closed and both cross-site protocols): all 31 subsets of the 5 groups; extra i.i.d.
  removal (p = 0.3) and pattern removal (transplanted from other test windows, p = 0.5).

## Verified locally (2026-11-07)
- 37 tests pass; the 5 Step 3 tests use synthetic Step 2 output for every 6th real window (type visible in the
  camera, weaker in audio, radar empty at V2/V3): loading keeps covered windows only; unknown types never
  reach train/val; dropping never adds a sensor or empties a window; AUROC/FPR are correct on known cases;
  an end-to-end run resumes without retraining and writes the report and tables.
- Whole notebook run locally (CPU, 2 seeds, 4 protocols, 5 epochs) on the synthetic data: every step
  succeeded; report, 4 tables and 4 figures written; 5.7 MB of runs.
