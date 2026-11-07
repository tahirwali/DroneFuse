# T1: importing TSMS-Drone into Kaggle

TSMS-Drone (Han & Jung, *Scientific Data* 13:407, 2026, doi:10.1038/s41597-026-06802-6; data on figshare,
doi:10.25452/figshare.plus.30027313, CC BY 4.0) is a second real, synchronised multi-sensor drone dataset:
an FMCW radar, a CW radar and an RF receiver record the same measurement at the same time. Targets: Inspire 2,
Mavic 2 Pro, Phantom 4 Pro, Matrice 30 and a corner reflector, hovering at 1.5 m height and 2-30 m away (2 m steps),
500 measurements per target and distance, in an open outdoor field.

## What is on figshare (checked 2026-11-07 through the figshare API and remote 7z headers)
- 85 files, 393 GB. `CW Radar.7z` (2.4 GB) and `RF Receiver.7z` (18.6 GB) each hold
  `<d>m/<target>/{Raw File,Image File}/<target>_<d>m_<idx>.{mat,png}` for all 75 target-distance pairs (37,500 + 37,500 entries).
- FMCW: 75 archives of about 5 GB, one per distance and target (`TSMS-Drone/Dataset/FMCW Radar/<d>m/<target>.7z`),
  each with 500 `.mat` and 500 `.png`.
- Format: 7z with LZMA2. The signed S3 links behind `ndownloader.figshare.com` expire after 10 s, so every
  request (including a resumed download) goes through the figshare link.

## What the notebook keeps
Only the processed 227 x 227 images (range-Doppler map for FMCW, Doppler spectrum for CW, power spectral density
for RF), packed as `CW_<d>m.tar`, `RF_<d>m.tar` and `FMCW_<d>m_<target>.tar`. Inside each tar: `<target>/<file>.png`.
The raw files stay on figshare.

## Running it
`notebooks/t1_tsms_import.ipynb`: CPU, Internet On, input `dronefuse-code`. Download speed decides the run time.
If the 11 h budget runs out, save the output as part 1, attach it, and run again; finished archives are skipped.
Save the result as `dronefuse-tsms`.
