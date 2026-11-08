# F1 — Every figure of the paper from the saved experiment outputs

**Kaggle:** Accelerator None (CPU), Internet Off is fine. About 20–30 min.
**Notebook:** `notebooks/f1_figures.ipynb`. **Code:** `dronefuse/figures.py`. **Tests:** `tests/test_figures.py`.

## What
One notebook draws all figures of the article from the files the earlier runs saved. It trains no new model
for MMAUD and types no number in. Each figure is written three times:

| File | Use |
|---|---|
| `<name>.pdf` | vector figure for the paper (fonts embedded) |
| `<name>.png` | quick look |
| `<name>.csv` | exactly the values plotted |

`figures_manifest.json` lists for each figure the input files it read (SHA-256; for files above 256 MB, of the
first and last 8 MB plus the size, marked `partial:`), the checks it passed, and the code version (one hash over
all modules of the package).

| Figure | Content | Read from |
|---|---|---|
| `fig_mmaud_recordings` | every MMAUD recording on its own time axis, ten blocks coloured train / validation / test | `splits/windows.csv` |
| `fig_examples_mmaud` | one window: camera frame, Avia, Mid-360 (static background in grey), radar, 6 s of sound | Step 1 extract (`V2_p4s1`) |
| `fig_examples_more` | DroneRFc-MM PTZ and fisheye frames, sound, radar; TSMS FMCW / CW / RF images of Mavic 2 Pro and Phantom 4 Pro at 10 m | T2 and T1 data |
| `fig_embedding_mmaud` / `_rfcmm` / `_tsms` | t-SNE maps coloured by drone type and by flight (or distance), with nearest-neighbour shares | Step 2 features, T3 feature cache |
| `fig_probes` | type probe vs flight / session probe per sensor, three datasets, random-guessing marks | Step 2 / 2b features (recomputed), `t3_results.json` |
| `fig_within_vs_cross` | MMAUD accuracy, same vs new flights, 95 % block-bootstrap intervals | Step 3 runs (bootstrap recomputed) |
| `fig_within_vs_cross_more` | the same for DroneRFc-MM and TSMS-Drone | `t3_results.json` |
| `fig_training_curves` | validation accuracy per epoch next to test accuracy of the same model | Step 3 run histories |
| `fig_confusion_mmaud` | confusion matrices, same flights vs new flights (DroneFuse, 3 seeds) | Step 3 saved logits |
| `fig_confusion_rfcmm` | the same on DroneRFc-MM (concatenation) | T3 feature cache (refit) |
| `fig_scores_rfcmm` | maximum class probability of known vs unknown windows, known from training vs new flights (Proposition 2) | T3 feature cache (refit) |
| `fig_loo_auroc` | AUROC of rejecting each left-out type, every method, all datasets; 50 = random | bootstrap + `t3_results.json` |

## Why
- Figures must come from the actual experiments, with a trail a reviewer can follow: the CSV holds each plotted
  value and the manifest says which file it came from.
- The MMAUD runs saved every test logit, so confusion matrices and score distributions need no retraining. Before
  drawing, each seed's rebuilt accuracy must equal the accuracy saved by Step 3, or the figure is not made.
- T3 did not save per-window probabilities, so for DroneRFc-MM the logistic regressions are refitted from the
  cached features with the T3 code. The figure is drawn only if the refit reproduces the saved accuracy and AUROC
  of every protocol (to the 0.1 % they were saved with); otherwise the step fails with both numbers.
- The nearest-neighbour shares make the t-SNE maps quantitative: of each window's 10 nearest neighbours (standardised,
  PCA-50 space), the share of the same type, and of those the share from the same flight. About 50 % means the
  features ignore the flight (two flights per type); about 100 % means they cluster by flight.

## How
- `find_inputs` locates each saved output anywhere under `/kaggle/input` (the Step 3 `runs/` folder with the most
  files, `*.deep.npz`, `*.crop.npz`, Step 1 `meta.json`, `t3_results.json`, the T3 cache marker, TSMS packs,
  `PTZ_meta.yaml`). A figure whose input is missing is skipped and the report says so.
- Each figure is one `StepRunner` step, so one failure never stops "Save & Run All".
- Style: Elsevier widths (90 / 190 mm), serif text, palette slots blue / orange / aqua (colour-blind checked),
  train / validation / test colours as in Figure 1, a blue–grey–red scale centred on 50 for AUROC.

## Inputs to attach
`dronefuse-code` (required), `dronefuse-step3-results`, `dronefuse-step2-features`, `dronefuse-step2b-crops`,
`dronefuse-mmaud-v2b`, the T3 output saved as a dataset (with `t3/cache/`), `dronefuse-rfcmm-core`,
`dronefuse-tsms`. If DroneRFc-MM and TSMS-Drone came from two T3 runs, attach both outputs: every
`t3_results.json` is read and merged per dataset (`load_t3`), the DroneRFc-MM cache is the one holding
`rfcmm_features.pkl`, the TSMS cache the one holding `tsms/*.npz`.

## Verified locally (2026-11-08)
- 7 tests (`tests/test_figures.py`); 68 pass in total. The refit test also checks that a changed saved value
  makes the step fail.
- On the real Step 3 runs and T3 results here: the rebuilt confusion matrices give 97.7 % (same flights), 32.0 %
  and 74.6 % (new flights), as in the paper, and every seed reproduces its saved accuracy.
- Whole notebook dry run on a fake `/kaggle/input` (real Step 3 runs and TSMS result, synthetic Step 1/2, DroneRFc-MM
  cache and TSMS packs): all 14 figures written, every step succeeded.

## First Kaggle run (2026-11-08)
- All steps succeeded; 13 of 14 figures written. Every MMAUD accuracy and 95 % interval equals Table `tab:xrec`;
  every seed reproduces its saved accuracy; the DroneRFc-MM refits equal the saved T3 accuracy and AUROC of all six
  protocol/method pairs.
- The attached T3 dataset (`dronefuse-t3-cache`) was an earlier T3 output with DroneRFc-MM only, so TSMS-Drone is
  missing from `fig_within_vs_cross_more`, `fig_probes` and `fig_loo_auroc`, and `fig_embedding_tsms` was skipped.
  Fixed by reading several T3 outputs (above); rerun with the later T3 output attached.
- Probes: the recording probe equals the paper in every row. Three type-probe cells differ by 0.1-0.2 from the
  paper table (Mid-360 all 62.1 vs 61.9, foreground 52.7 vs 52.8, crop maximum pooling 56.5 vs 56.6); the plain
  accuracies equal the Step 2 log, so the paper table carries small transcription or library-version differences.
