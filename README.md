# DroneFuse

Multimodal drone recognition that keeps working when sensors drop out, and that can say
"I don't know this drone" instead of guessing.

Counter-UAV sites combine cameras, LiDAR, radar and microphones, but in practice some of these miss
the drone much of the time: it flies out of a LiDAR's field of view, it is too small for the radar,
or the wind covers the audio. A deployed system will also eventually meet a drone type it was never
trained on. DroneFuse studies both problems together on the synchronised multi-sensor dataset
[MMAUD](https://doi.org/10.1109/ICRA57147.2024.10610957).

Everything runs on Kaggle's free tier: the extraction steps need only the CPU, and the two GPU steps
fit in a few hours on a T4.

## Pipeline

| Step | Notebook | What it does | Hardware |
|---|---|---|---|
| 0 | `step0_mmaud_probe.ipynb` | Reads the bag indexes on the Hugging Face Hub: topics, rates, formats, download speed | CPU |
| 1 | `step1_extract_{v1,v2a,v2b}.ipynb` | Streams all 12 recordings (~373 GB of bags) and keeps a compact 8.5 GB copy of every sensor | CPU |
| 2 | `step2_features.ipynb` | Per-window sensor availability, point-cloud and audio descriptors, frozen ConvNeXt-T and AST embeddings | GPU |
| 2b | `step2b_camera_crops.ipynb` | Finds the drone in each camera frame without labels and embeds a small crop around it, so the camera can't recognise the recording from the sky | GPU |
| 3 | `step3_fusion.ipynb` | Trains every method on every protocol (3 seeds), unknown-drone rejection, sensor-removal tests, paper tables and figures | GPU |
| T1 | `t1_tsms_import.ipynb` | Imports the second dataset, TSMS-Drone (FMCW radar, CW radar, RF receiver), straight from figshare into Kaggle and keeps its processed images | CPU |
| T2 | `t2_dronerfcmm_import.ipynb` | Imports the third dataset, DroneRFc-MM (PTZ and fisheye camera, RF, LiDAR, mmWave radar, microphones), from Science Data Bank into Kaggle in parts of at most 15 GB | CPU |
| T3 | `t3_extra_datasets.ipynb` | Runs the recording check (probes, within vs cross-session accuracy, unknown-type rejection) on TSMS-Drone and DroneRFc-MM | GPU T4 |

The `docs/` folder explains each step in more detail.

## Data and splits

The windows are 1 s long. Splits are made from time blocks within each recording (never from random
frames), and every choice (thresholds, best epoch, rejection cut-off) is made on the validation
split. The fixed windows and protocols are in `splits/`:

- `closed`: all six drone types known
- `v1_to_v2` and `v2_to_v1`: train at one site, test at the other
- `loo_<type>`: leave one drone type out and treat it as unknown
- `xrec_s1_to_s2` and `xrec_s2_to_s1`: train on one car-park flight per type and test on the other flight

## Methods compared

Single-sensor models, late fusion, feature concatenation, an LMMUAVC-style model (a learned slot for
each missing sensor, with random sensor dropping), and DroneFuse. DroneFuse uses attention that skips
missing sensors, and during training it removes sensors in the patterns that actually co-occur in the
data. We score unknown drones with max-softmax, energy, and an energy score normalised by how many
sensor groups saw the window.

## Running it

1. Zip this folder and upload it to Kaggle as a dataset named `dronefuse-code`.
2. Import a notebook from `notebooks/`, set the accelerator named at its top, and add `dronefuse-code`
   plus the dataset produced by the previous step as inputs.
3. Use **Save Version → Save & Run All**. Each notebook finds its inputs by itself, and it finishes
   and saves its report even if one step fails.

To run the tests locally (no GPU or internet needed):

```
pip install -r requirements.txt
python -m pytest tests -q
```

## Layout

```
dronefuse/   the Python package (extraction, features, fusion, evaluation)
notebooks/   one Kaggle notebook per step
splits/      fixed windows, splits and protocols
tests/       offline tests on small synthetic bags and features
docs/        notes on each step
```

## Licence

The code is released under the MIT licence (`LICENSE`). MMAUD is licensed CC BY-NC-SA 4.0, so
anything derived from it (the files in `splits/` and any features or results produced by these
notebooks) is released under CC BY-NC-SA 4.0 as well (`splits/LICENSE`). The MMAUD data itself is not
included here. It is available from the Hugging Face Hub (`Yuan-Shenghai/MMAUD`). The pretrained
models (timm `convnext_tiny.fb_in1k`, `MIT/ast-finetuned-audioset-10-10-0.4593`) are downloaded when
the code runs and keep their own licences.

If you use the dataset, please cite MMAUD:

> S. Yuan et al., "MMAUD: A Comprehensive Multi-Modal Anti-UAV Dataset for Modern Miniature Drone
> Threats," ICRA 2024. doi:10.1109/ICRA57147.2024.10610957
