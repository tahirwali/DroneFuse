# Step 2 — Per-window sensor availability and frozen features

## What
One GPU notebook (`notebooks/step2_features.ipynb`) reads the Step 1 datasets `dronefuse-mmaud-{v1,v2a,v2b}`
and computes, for every window of `splits/windows.csv`:

| Output | Content | File |
|---|---|---|
| availability | per window: camera frames; Avia / Mid-360 / radar points, all and foreground; radar tracks; audio frames per node | `<rec>.npz` |
| covered | window inside the extracted time range of every stream (an extraction stopped by the time budget leaves later windows uncovered) | `<rec>.npz` |
| point-cloud descriptors | Avia, Mid-360, radar: 39-dim (LiDAR) / 49-dim (radar) descriptor of all points and of foreground points | `<rec>.npz` |
| point sets | up to 512 points (LiDAR) / 256 (radar) per window, with mask: Avia all points, Mid-360 and radar foreground | `<rec>.npz` |
| audio | log-mel 64×101 at 16 kHz per node; mean+std over time (128-dim) per node | `<rec>.npz` |
| camera embeddings | ConvNeXt-T (ImageNet-1k), native 1280×960, last feature map average- and max-pooled, mean over the window's frames (768 + 768) | `<rec>.deep.npz` |
| audio embeddings | AST fine-tuned on AudioSet, pooled output (768) per node | `<rec>.deep.npz` |

It then trains one logistic regression per sensor block on the **closed-protocol train split** and reports
accuracy on the **validation split only** (`step2_report.txt`). The test splits are not used in Step 2.

## Why
- **Availability is the core of the paper's realism claim.** LMMUAVC removes sensors independently at
  random. In MMAUD, whether a sensor sees the drone depends on where the drone is (field of view, range), so
  sensors fail together. Counting what each sensor delivered gives the *natural* missingness pattern of
  every test window. Counts are not thresholded here: what counts as "present" is chosen on validation in
  Step 3.
- **Static background removal.** Raw LiDAR and radar clouds are dominated by the scene. Checked on 36 s of
  V1_Mavic3, the radar's median range stays at 6.6 m whatever the drone does. A voxel occupied in ≥ 10 % of
  a recording's scans is background. This uses no labels or ground truth, like a fixed counter-UAV site
  that maps its surroundings first.
- **Frozen backbones keep the GPU cost small** (Kaggle quota) and the comparison with LMMUAVC fair (same
  ConvNeXt-T). Max pooling is added because the drone covers only a few pixels: average pooling describes
  the scene, while max pooling can keep a strong local response.
- **Ground truth is never a feature.** The Leica "relative" position is not in any sensor's frame (Avia,
  Mid-360 and radar points are not near it; checked on the Mavic3 slice), so it could not be used to crop
  the drone without an extrinsic calibration that the release does not provide.

## How
- `dronefuse/features.py`: CPU part, numpy only (one process per recording).
  - Voxel keys are packed into int64. Background is the set of voxels with occupancy ≥ 10 % of scans;
    foreground is tested with `searchsorted`.
  - Descriptor: log count, fraction of scans with points, xyz mean/std, range mean/std/min/max, elevation
    mean/std, azimuth circular mean and concentration, sqrt eigenvalues of the covariance, centroid
    velocity (second half − first half of the window), 8-bin log range histogram, 6-bin elevation
    histogram, then reflectivity (LiDAR) or Doppler + Power (radar) mean/std and an 8-bin Doppler histogram.
  - Audio: every MMAUD audio message is exactly one 1152-sample MP3 frame (checked: 1,513 messages decode
    to 1,513 × 1152 samples). Each node is decoded once as one stream. Message *i* starts at sample
    *i*·1152, and a window's start sample is interpolated from message times. The two channels of a node
    are identical (correlation 1.0), so the signal is averaged to mono. 48→16 kHz uses a windowed-sinc
    low-pass (97 taps, 7.2 kHz) and keeps every 3rd sample. Log-mel: 25 ms Hann window, 10 ms hop, 64 bands
    from 50 Hz to 8 kHz.
- `dronefuse/deep.py`: GPU part (torch, timm `convnext_tiny.fb_in1k`, transformers
  `MIT/ast-finetuned-audioset-10-10-0.4593`), fp16 autocast.
- `dronefuse/step2.py`: driver, `find_recordings` (any `meta.json` under `/kaggle/input`), linear probe and
  report.

## Verified locally (2026-11-05)
- 31 tests pass. The 7 Step 2 tests use a synthetic recording: the static wall is removed and the moving
  point is kept; velocity +2.0 m/s is recovered; radar Doppler and power are recovered; empty windows give
  zero descriptors; coverage is false outside the extraction; MP3 tones land in the right windows (±20 Hz);
  the linear probe uses train and val only.
- Real V1_Mavic3 slice (36 s, 58 covered windows): CPU features in 23 s. Avia is empty in 69 % of windows.
  Mid-360 foreground has a median of about 150 points per window, from about 42,000 points before removal.
  Radar foreground is empty in 78 % of windows. Every audio node delivers 41–42 frames per window.
- The full notebook was run locally (CPU torch) on the same slice.

## Findings so far
- 36 s of data is too little to judge whether the foreground is the drone, since people and vehicles
  also move. Step 3's per-sensor results on validation will tell.
