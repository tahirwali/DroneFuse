# Step 0 — Probe of the MMAUD dataset

## What
A CPU-only Kaggle notebook (`notebooks/step0_mmaud_probe.ipynb`) that finds out what MMAUD really contains,
in what format, and what our Kaggle sessions can handle, before any extraction or training code is written.

## Why
MMAUD is ~373 GB on the Hugging Face Hub (`Yuan-Shenghai/MMAUD`). Its documentation leaves open the
facts that decide the design of the whole pipeline:
- which sensor topics each recording has, at what rate, for how long (V1 vs V2/V3 differ);
- how audio is stored (it is **not** in the extracted V1 files; it lives only inside the bags);
- how big each sensor stream is, so we know what must be downloaded;
- how much RAM, disk and download speed a Kaggle session really gives.

## How
- **Bag index over HTTP.** A ROS1 bag (format 2.0) stores its index at the end of the file: one record
  per connection (topic, message type, message definition) and one per chunk (file position, time range,
  messages per connection). `dronefuse/rosbag_index.py` reads the header, then only the bytes from
  `index_pos` to the end, using HTTP range requests (`dronefuse/remote.py`). About 1 MB per bag gives
  every topic's count, duration and rate. The 54.7 GB `m3e.bag`, delivered as three byte parts, is read
  as one virtual file.
- **Sample messages.** A few chunks per bag (one per topic, nearest the middle, plus a few spread out) are
  downloaded, bz2-decompressed and decoded with `rosbags`, using the message definitions stored in the bag.
  Mean message size × count gives each topic's total size.
- **Audio.** Consecutive chunks are read until ~5 s of every audio topic is covered; the payloads are joined
  and decoded with `soundfile`.
- **Every step is wrapped** by `dronefuse/runner.py`: an error is printed and recorded, and the notebook
  continues, so "Save & Run All" always finishes and saves `probe_summary.json` and `probe_report.txt`.
  Large downloads go to a scratch folder outside `/kaggle/working` and are deleted at the end.

## Already verified from the local machine (2026-11-03)
`V1/rosbag/Mavic3.bag` (11.88 GB): index read in 7 s (1.0 MB). 198.4 s long, 5,979 chunks (~2 MB each).
| Topic | Type | Messages | Rate | Size per message |
|---|---|---|---|---|
| `/usb_cam/image_raw` | sensor_msgs/Image, **rgb8 2560×960 (left+right side by side)** | 5,945 | 30 Hz | 7.37 MB raw (~44 GB uncompressed) |
| `/livox_avia/lidar` | livox_ros_driver/CustomMsg, 24,000 points | 1,984 | 10 Hz | 456 kB |
| `/livox_mid360/lidar` | livox_ros_driver2/CustomMsg, 19,968 points | 1,908 | 9.6 Hz | 379 kB |
| `/radar_enhanced_pcl` | sensor_msgs/PointCloud (~370 pts; channels incl. Doppler) | 2,975 | 15 Hz | 11 kB |
| `/radar_pcl` | sensor_msgs/PointCloud (~12 pts) | 2,976 | 15 Hz | 0.5 kB |
| `/radar_trk` | sensor_msgs/PointCloud (tracks; VeloX…) | 2,975 | 15 Hz | 0.2 kB |
| `/audio1..4/audio` | audio_common_msgs/AudioData | 8,266 each | 41.7 Hz | ~0.3 kB |

Audio payload = **MP3, 48 kHz, 2 channels per node, ~100 kbit/s** (frame sync `fffb`; decoded with soundfile).
Ground truth bag: `/leica/point/absolute` and `/relative` (geometry_msgs/PointStamped) at 4.9 Hz.

### Full local dry-run of the notebook (2026-11-04, all 37 steps succeeded)
- Hub inventory: **83,488 files, 372.7 GB** (V2/V3 bags 193.6 GB, V1 bags 77.9 GB, V1 PNG images 73.5 GB).
- **All 12 sensor bags have the same 10 sensor topics** (4 audio @ 41.7 Hz, Avia @ 10 Hz, Mid-360 @ ~9.6 Hz,
  3 radar @ 15 Hz, camera @ 30 Hz); chunks ≈ 2 MB everywhere; bz2 ratio ≈ 3.6–3.8.
- Durations: V1 Avata 354 s, M300 257 s, Mavic2 252 s, Mavic3 198 s, Pham4 237 s (**1,298 s ≈ 21.6 min**);
  V2/V3 m300s1 234 s, m300s2 361 s, m30s1 467 s, m30s2 202 s, m3e 832 s, p4s1 430 s, p4s2 474 s (**3,000 s = 50 min**).
- Ground truth: V1 `/leica/point/*` PointStamped ≈ 4.9 Hz; V2/V3 `/leica/pose/*` PoseStamped ≈ 16–21 Hz.
- V1 extracted files: image PNG **2560×960 RGB** (stereo pair, 2.2 MB); lidar_360 `(≈20k, 3) float64`;
  livox_avia `(24000, 3) float64` — **the first files are all zeros** (check how many frames are empty);
  radar_enhance_pcl `(≈333, 3) float64` (xyz only; Doppler/other channels are in the bag, not in the NPY);
  ground_truth `(3,) float64` at ≈4 Hz.
- V2/V3 WAV: 48 kHz, **2 channels**, PCM-16, full recording length (201.6 s for m30s2), ~39 MB per node.
- Background noise: `(48000, 4) float32` = 1 s × 4 channels.
- Bag audio decodes as MP3 48 kHz stereo on every node, V1 and V2/V3. Decoded length differs slightly from bag time
  (5.7 s of samples for 5.26 s of bag time; node 3 shorter) → align audio by **sample count from a known start**, not by message stamps.
- Download speed from this PC: 2.6–6.3 MB/s (Kaggle to be measured).

### Dataset structure that matters for the experiment design
| Drone type | V1 (rooftop) | V2/V3 (carpark) |
|---|---|---|
| DJI Avata | ✓ | – |
| DJI M300 | ✓ | m300s1, m300s2 |
| DJI Mavic 2 | ✓ | – |
| DJI Mavic 3 / Mavic 3 Enterprise | Mavic3 | m3e |
| DJI Phantom 4 | Pham4 | p4s1, p4s2 |
| DJI M30 | – | m30s1, m30s2 |

Each recording is one flight of one drone type, so **class = recording**; splits must be time blocks, and
cross-scene tests (V1 ↔ V2/V3) are needed to show the model learns the drone, not the session.
Types seen in only one version (Avata, Mavic 2, M30) are natural open-set "unknown" candidates.

### Kaggle run (2026-11-05, CPU session, all steps succeeded)
- 4 CPUs, 32.9 GB RAM, no GPU; `/kaggle/working` 21 GB free; **`/tmp` 1,038 GB free**; **HF → Kaggle 75.3 MB/s**.
- Inventory, bag indexes, decoded messages and audio identical to the local dry-run.
- V1 ground-truth bags are shorter than the sensor bags (Avata 333.8 s vs 354.2 s; Mavic3 184.6 vs 198.4 s).
- V2/V3 `/livox_mid360/lidar` ≈ 761 kB/msg (V1: 379 kB) for the same 19,968 points — point format to check.
- V2/V3 `/radar_pcl` can be empty; `/radar_enhanced_pcl` holds 59–370 points with Doppler.

Consequences for Step 1 (confirmed by the Kaggle run):
- Audio is ~0.01 GB per recording but is spread through every chunk, so getting V1 audio means reading the
  whole bag once (12–21 GB each). The bag does **not** need to be stored: it can be streamed chunk by chunk over
  HTTP, each ~2 MB chunk decompressed, the wanted topics kept (audio; downscaled frames for V2/V3) and the rest
  discarded. Disk use then stays at a few GB regardless of bag size; download speed sets the time.
- Images dominate the size; V1 already has them as PNG files on the Hub, so no image decoding from bags is needed for V1.
