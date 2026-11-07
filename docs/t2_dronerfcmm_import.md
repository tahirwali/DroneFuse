# T2 - Import DroneRFc-MM into Kaggle

**Notebook:** `notebooks/t2_dronerfcmm_import.ipynb` | **Code:** `dronefuse/rfcmm.py` | **Tests:** `tests/test_rfcmm.py`
**Kaggle:** Accelerator None (CPU), Internet On, input `dronefuse-code`.

## The dataset

DroneRFc-MM (Yu, Yang, Hu, Li, Wu, Su, Pan and Shi; Science Data Bank, doi:10.57760/sciencedb.j00173.00094,
CC BY-SA 4.0, version 1.0.0). Dataset paper: Journal of Electronics & Information Technology, under review.

- Sensors: PTZ camera, fisheye camera, RF receiver (complex IQ), LiDAR, mmWave radar, four-channel microphone
  array. Soft synchronisation by network time, about 0.3 s offset between devices. Flight-log ground truth
  (position, speed, attitude).
- Sessions A1, A1-2, B1, C1, E1, E1-2, F1, G1, G1-2, all on 2026-04-01 between 13:46 and 15:46 at one site,
  3-7 min each. A1 = DJI Mavic 3, B1 = DJI Avata (no RF: its band cannot be fixed). A1, E1 and G1 were flown a
  second time about 1.5 h later, a separate take-off, so a cross-recording test is possible for three types.
- 113 files, 75.6 GB: `raw_data/` (GT csv, PTZ and fisheye mp4, audio .dat, LiDAR .pcd 18.8 GB, radar zip),
  `RF_raw/*.rar` (43.8 GB), `samples/` (5 s clips as videos with flight-direction labels), `directionQA/` code.

## Access

No login. File list: `POST https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath`
with `{dataSetId, version: "V1", path: "/V1/...", lastIndex, pageSize}`. Download:
`https://china.scidb.cn/download?fileId=<id>`, which supports range requests. One connection gives about
0.3 MB/s; 16 parallel range requests gave 3.5 MB/s (measured from China on 2026-11-07).

## What the notebook does

A notebook can save about 20 GB, so the release is split into parts of at most 15 GB (`plan_parts`):

| Part | Files | Size |
|---|---|---|
| `core` | everything except LiDAR and RF raw data | 12.97 GB |
| `lidar_1` | LiDAR A1-2, A1, B1, C1, E1-2, E1 | 12.40 GB |
| `lidar_2` | LiDAR F1, G1-2, G1 | 6.43 GB |
| `rf_1` ... `rf_4` | two RF archives each | 9.3, 12.0, 12.5, 10.0 GB |

Set `PART` in the settings cell. The notebook measures the speed, then downloads each file of the part with
`WORKERS` parallel range requests into `/kaggle/working/rfcmm/<release path>`, checks the MD5 and keeps the
file unchanged. Interrupted files resume from their finished chunks within a run; files already in an
attached earlier run of the same part are skipped. Files are not unpacked (the RF archives are .rar; they will
be read by the feature step).

## After the run

Send `rfcmm/t2_report.txt`. Save the output as `dronefuse-rfcmm-<part>`. If files are *not reached* or
*failed*, attach that dataset and run the same part again.

## Checks done

- Offline tests: file-list walk with paging, part packing under the limit, chunked download with resume and
  MD5 check, skip of files already present, time budget.
- Live, from China: the full file list (113 files, 75.61 GB), two real files downloaded with matching MD5, the
  speed test, and a dry run of the notebook cells on five small files, twice (second run skipped everything).
- Not yet: a full part on Kaggle (speed from Kaggle unknown).
