# Step 1 — Extraction of all 12 MMAUD recordings, and the fixed splits

## What
Three CPU-only Kaggle notebooks (`notebooks/step1_extract_{v1,v2a,v2b}.ipynb`) turn the 271 GB of official
MMAUD sensor bags into a compact, time-indexed copy of every sensor (≈9 GB in total), saved as three private
Kaggle datasets `dronefuse-mmaud-{v1,v2a,v2b}`. The train/val/test windows were fixed beforehand from the bag
indexes alone and are committed in `splits/` (see `splits/README.md`).

## Why
- The bags hold 2560×960 raw images at 30 Hz (~44 GB per recording uncompressed); training cannot read them.
- Every sensor, including audio and radar Doppler, is in the bags; the extracted V1 files on the Hub lack audio
  and Doppler, and the Kaggle V1 mirror misfiles ~2,021 Avata frames as Phantom 4 (Session 6). One pipeline from
  the official bags gives the same processing for V1 and V2/V3.
- Windows and splits depend only on time, so fixing them before any model exists rules out tuning the split.

## How
- **Streaming.** A bag is cut into tasks of 32 consecutive chunks (~64 MB). Each worker process (one per CPU)
  downloads its task with one HTTP range request, decompresses the bz2 chunks and decodes the wanted messages
  (`dronefuse/extract.py`). Nothing is written to disk except the kept payloads.
- **Decoding.** `dronefuse/ros1.py` decodes messages from the definitions stored in the bag; arrays of
  fixed-size elements (LiDAR points, Point32, image bytes) become numpy views, so a 24,000-point scan costs
  microseconds. Checked against `rosbags` in the tests.
- **What is kept** (positions within each topic decide what is kept, so the output does not depend on how the
  bag is cut into tasks — tested):

  | Stream | Source topic | Kept | Stored as |
  |---|---|---|---|
  | camera | `/usb_cam/image_raw` | every 3rd frame (10 Hz) | left half, native 1280×960, JPEG q92 (~143 kB) |
  | livox_avia, livox_mid360 | `/livox_*/lidar` | every 2nd scan (5 Hz) | non-zero points: float32 x, y, z + uint8 reflectivity |
  | radar | `/radar_enhanced_pcl` | all (15 Hz) | float32 x, y, z, Doppler, Range, Power, Alpha, Beta |
  | radar_tracks | `/radar_trk` | all (15 Hz) | float32 x, y, z, VeloX, VeloY, VeloZ, Class, Confidence |
  | audio1–4 | `/audio{1..4}/audio` | all (41.7 Hz) | original MP3 frames (48 kHz stereo per node) |
  | ground truth | `/leica/point|pose/{relative,absolute}` | all | `ground_truth.npz` |

  Format: per stream a `.bin` file of payloads and a `.index.npz` (bag time, header stamp, offset, length),
  sorted by bag time; `meta.json` describes every stream (`dronefuse/store.py`).
- **Safety for Save & Run All.** One step per recording; a failed download range is retried once, then recorded;
  decoding always runs in worker processes, never in the notebook's process. Each worker is replaced by a fresh
  ("spawn") process after 40 tasks (~2.7 GB of bag), so memory cannot build up. If a worker dies anyway (out of
  memory), a new pool with one worker fewer takes over the unfinished tasks (at most 3 times per recording; after
  that the remaining ranges are recorded as failed and the recording is marked incomplete). Progress lines show
  the container's memory in use. No new work starts in the last 15 min of the 11 h 15 min budget, and an
  unfinished recording is marked `STOPPED BY TIME BUDGET`.
  *(Changed 2026-11-06 after the v2b run was killed for lack of memory.)*
- **bz2 speed.** Decompression is the bottleneck (local PC: 0.9–1.9 MB/s of compressed data per core). Each
  notebook times the standard `bz2` module and `indexed_bzip2` on real chunks and uses the faster one only if
  its output is byte-identical.

## Verified locally (2026-11-05)
- 24 tests pass (`python -m pytest tests -q`), including 2-process extraction = 1-process extraction.
- Real slices of `Mavic3.bag` and `m30s2.bag`: all 9 streams decode at the expected rates (camera 10.0 Hz,
  LiDAR 5.0 Hz, radar 15.0 Hz, audio 41.7 Hz); every audio payload starts with an MP3 frame sync; 3.14 s of
  bag time decodes to 3.19 s of 48 kHz stereo audio.
- Notebook dry-run (V1_Mavic3, 5-min budget): every step succeeded; stopped cleanly at the deadline after
  1,088 of 5,979 chunks; ≈0.44 GB projected for the whole recording.

## Findings that matter for the paper
- **The camera is a sky-facing fisheye; the drone is a ~5-pixel speck** in a 1280×960 half. Whole-frame image
  features mostly describe the scene (rooftop vs carpark, clouds, sun), not the drone. This supports the
  cross-scene protocol and is a likely reason why within-recording splits look easy.
- **Livox Avia returns very few points** (0–431 non-zero points per 24,000-point scan; 0 in the first seconds)
  — it sees the sky, so its returns are mostly the drone itself, and absent when the drone is outside its field
  of view. **V2/V3 radar is often empty** (0–60 points per scan). MMAUD therefore has *natural*, structured
  missing modalities; Step 2 will measure per-window sensor availability and use it as a realistic missingness
  pattern, next to synthetic masking.
- Mid-360 messages carry a variable number of points (20k–80k declared), hence the 2× larger V2/V3 messages
  seen in Step 0; ~23–44 % are non-zero.
- Ground-truth bags cover less than the sensor bags (V1 Avata: 331 of 354 s usable); windows are restricted to
  tracked time.
