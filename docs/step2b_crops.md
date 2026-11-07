# Step 2b: drone-centred camera crops, and the recording shortcut

## Why
In Step 2 a linear probe on the whole-frame camera embeddings reached 100 % on validation, both pooling
variants. The drone covers only a few pixels of the sky-facing fisheye image, so this can't come from the
drone. Each drone type has only one or two flights per site, and validation blocks come from the same flights
as training, so the probe can recognise the **recording** instead: that day's clouds and light, and the
buildings. The Mid-360 points the same way (all points 0.647 > foreground 0.524).

We do two things about it:
1. **Measure it.** New protocols `xrec_s1_to_s2` and `xrec_s2_to_s1` (see `splits/README.md`) train on one
   car-park flight of M300, M30 and Phantom 4 and test on the other flight. The Mavic 3E is the unknown type.
   Known and unknown test windows then both come from unseen recordings. A recording probe (Step 2b
   report) asks directly how well each sensor tells flight 1 from flight 2 of the same type (chance 50 %).
2. **Remove it from the camera.** Embed only a crop around the drone, so the scene is mostly cut away.

## How (`dronefuse/crops.py`, notebook `step2b_camera_crops.ipynb`)
- Black-hat filter (15 px closing minus the image) on max(R, G, B). The drone is dark in every channel,
  while blue sky is bright in blue. On grey-scale the gaps between clouds looked like dark spots.
- Subtract the larger of (a) the median response over 120 frames spread across the recording, which removes
  tree tops, building edges and the lens rim, and (b) the response near the same place 3 frames (0.3 s)
  earlier and later, which removes cloud gaps that barely move. Search only inside a sky mask (bright in the
  median image, eroded by 25 px).
- Per window: the 3 frames with the strongest response. A 128 px crop around each is resized to 224,
  standardised per colour channel (so sky brightness and colour carry no information) and embedded by the
  same ConvNeXt-T as Step 2 (avg and max pooled). The peak strength is kept as an extra feature: it says how
  clearly the camera sees a drone.
- The filters run on the GPU with max-pooling; CPU workers only decode JPEGs.
- Step 3 runs everything a second time with `camera="crop"`; runs go to `runs_crop/`, tables to
  `tables_crop/`. In that variant the camera is **present only where the drone was seen** (peak >= 20,
  `crops.DETECTED_PEAK`); other windows get no camera input, so crops of empty sky never reach the model.

## Checked locally (2026-11-07) on real frames from the Hub
- V2_p4s1 (car park, overcast): frames from 12 short segments. Windows with the drone in view have peaks
  of 67–139 and the crop is centred on the drone. The others stay at 54 or below: drone too small or out of the
  sky region.
- V1_Mavic2 (rooftop, blue sky with cumulus): on grey-scale the cloud gaps won. With max(R, G, B) the drone
  is found in some windows, including small ones. The local sample has only ~5 frames per segment, so the
  motion check could not be tested fairly there. **The contact sheets of the Kaggle run must be checked by eye.**
- Tests: `tests/test_step2b.py`. The cross-recording protocols never test on a training flight; the locator
  finds a moving 4 px drone and ignores a static dark spot and an edge "tree"; standardised crops are
  independent of overall brightness; the crop variant trains into `runs_crop/` and resumes.

## Limits
A hovering drone is partly suppressed by the motion check. A drone in front of a building, outside the sky
mask, is not searched for. Windows where the drone is not visible still get a crop (of sky), with a low
peak value.

## Result of the Kaggle run (2026-11-07)
- Contact sheets: when the drone is in view the crop is centred on it (peak ~45-200; faint drones on overcast sky 12-18).
  Otherwise the crop is empty sky (peak 4-10). Median peak is 4-8 in every recording, so the drone is seen in a minority of windows
  (V1_Mavic2 almost never). Occasional false detections: the lens rim, a building edge, a person's head.
- Linear probe on drone type (val): whole frame 1.000, crop (avg) 0.736.
- Recording probe (chance 0.5): whole frame 1.000, crop (avg) 0.892, Mid-360 all 0.948 / foreground 0.894, Avia 0.714,
  audio log-mel 0.657 / AST 0.581, radar 0.552. The shortcut is real and not only in the camera. Empty-sky crops still
  carry the day's clouds, hence the threshold above. Step 3 repeats both probes on the "drone seen" windows only
  (report section SHORTCUT CHECK).
