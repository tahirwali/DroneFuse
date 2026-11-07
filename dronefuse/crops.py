"""
Drone-centred camera features (Step 2b).

The whole-frame camera embeddings of Step 2 let a classifier recognise the recording (the clouds, the
light, the buildings) instead of the drone, which covers only a few pixels of the sky-facing fisheye.
Here we look for the drone itself and embed only a small crop around it.

The drone is a small dark speck against the sky. We apply a black-hat filter (grey closing minus the
image) to the brightest colour channel: blue sky is dark in grey-scale but bright in blue, while the drone
is dark in every channel. Two things are then subtracted from the response:
  - what is always there (tree tops, building edges, the rim of the lens): the median response over frames
    spread across the whole recording, plus a sky mask of pixels that are bright in the median image;
  - what barely moves (gaps between clouds): the response at the same place a few frames earlier and later.
A flying drone moves several pixels in that time, so it survives both. No labels or ground truth are used.

For each frame the strongest remaining response inside the sky is taken as the drone. Each window keeps
the three frames where that response is strongest. Their crops are standardised per colour channel, so
overall brightness and sky colour carry no information, and embedded with the same ConvNeXt-T as Step 2.
The filters run on the GPU; the CPU workers only decode JPEGs.
"""
import io
from pathlib import Path

import numpy as np

from .deep import CAMERA_MODEL, device

FILTER = 15             # black-hat size in pixels: larger than a distant drone, smaller than clouds
NEIGHBOUR = 3           # frames before and after (0.3 s at 10 Hz) used to suppress things that barely move
SLACK = 5               # a speck may shift this many pixels between those frames and still count as static
CROP = 128              # crop side in native pixels, resized to INPUT for the backbone
INPUT = 224
MODEL_FRAMES = 120      # frames used to build the per-recording background model
SKY_MARGIN = 25         # pixels removed from the edge of the sky mask
BEST_FRAMES = 3         # frames per window that are embedded


def decode(payload):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))


def blackhat(rgb):
    """(B, H, W) black-hat response of the brightest channel. rgb: (B, H, W, 3) uint8 tensor."""
    import torch.nn.functional as F
    v = rgb.amax(-1).float().unsqueeze(1)
    pad = FILTER // 2
    dilated = F.max_pool2d(F.pad(v, (pad,) * 4, mode="replicate"), FILTER, stride=1)
    closed = -F.max_pool2d(F.pad(-dilated, (pad,) * 4, mode="replicate"), FILTER, stride=1)
    return (closed - v)[:, 0]


def background_model(reader, frames, dev, n=MODEL_FRAMES):
    """Median black-hat response and sky mask of one recording, from `n` frames spread over `frames`."""
    import torch
    import torch.nn.functional as F
    frames = np.unique(np.asarray(frames))
    pick = frames[np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int))]
    values, responses = [], []
    for start in range(0, len(pick), 16):
        rgb = torch.from_numpy(np.stack([decode(reader.payload(int(f))) for f in pick[start:start + 16]])).to(dev)
        values.append(rgb.amax(-1).half().cpu())
        responses.append(blackhat(rgb).half().cpu())
    median_value = torch.cat(values).float().median(0).values
    static = torch.cat(responses).float().median(0).values
    smooth = F.avg_pool2d(median_value[None, None], 31, stride=1, padding=15)[0, 0]
    sky = smooth > 0.6 * torch.quantile(median_value.flatten()[::16], 0.95)
    if SKY_MARGIN:
        sky = -F.max_pool2d(-sky.float()[None, None], 2 * SKY_MARGIN + 1, stride=1, padding=SKY_MARGIN)[0, 0] > 0.5
    if sky.sum() < 1000:            # no usable sky (should not happen with this camera): search everywhere
        sky = torch.ones_like(sky)
    return {"static": static.to(dev), "sky": sky.to(dev), "frames_used": len(pick)}


def locate(rgb, model):
    """(B, 3) tensor of x, y, peak: the strongest moving speck inside the sky in each of B consecutive frames."""
    import torch
    import torch.nn.functional as F
    bh = blackhat(rgb)
    near = F.max_pool2d(bh.unsqueeze(1), 2 * SLACK + 1, stride=1, padding=SLACK)[:, 0]
    b = len(bh)
    steady = torch.zeros_like(bh)
    for i in range(b):
        others = [j for j in (i - NEIGHBOUR, i + NEIGHBOUR) if 0 <= j < b]
        if others:
            steady[i] = near[others].amin(0)
    score = bh - torch.maximum(model["static"].unsqueeze(0), steady)
    score = F.avg_pool2d(score.unsqueeze(1), 3, stride=1, padding=1)[:, 0]
    score[:, ~model["sky"]] = -1e9
    flat = score.flatten(1).argmax(1)
    w = score.shape[2]
    peak = score.flatten(1).gather(1, flat[:, None])[:, 0]
    return torch.stack([(flat % w).float(), (flat // w).float(), peak], 1)


def crop_at(rgb, x, y, size=CROP):
    h, w = rgb.shape[:2]
    x0 = int(np.clip(x - size // 2, 0, w - size))
    y0 = int(np.clip(y - size // 2, 0, h - size))
    return rgb[y0:y0 + size, x0:x0 + size]


def standardise(crops):
    """(B, 3, INPUT, INPUT): resized, zero mean and unit spread per crop and channel. crops: (B, S, S, 3) uint8."""
    import torch.nn.functional as F
    x = crops.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(INPUT, INPUT), mode="bilinear", align_corners=False)
    return (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 0.05)


class _Frames:
    """torch Dataset of decoded RGB frames (uint8), in the order given."""

    def __init__(self, folder, indices):
        self.folder, self.indices = folder, indices
        self.reader = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        import torch
        from .store import StreamReader
        if self.reader is None:
            self.reader = StreamReader(self.folder, "camera")
        return torch.from_numpy(decode(self.reader.payload(int(self.indices[k]))).copy())


def crop_embeddings(folder, frame_ranges, batch_size=32, workers=4, log=print):
    """
    Per window: the mean ConvNeXt embedding (avg and max pooled) of the crops of its BEST_FRAMES frames
    with the strongest speck, the strongest peak, and that frame's index and speck position.
    """
    import timm
    import torch
    from .store import StreamReader
    n = len(frame_ranges)
    needed = np.unique(np.concatenate([np.arange(a, b) for a, b in frame_ranges] or [np.zeros(0, np.int64)]))
    out = {"camera_crop_avg": np.zeros((n, 768), np.float32), "camera_crop_max": np.zeros((n, 768), np.float32),
           "camera_crop_peak": np.zeros(n, np.float32), "camera_crop_frame": np.full(n, -1, np.int64),
           "camera_crop_xy": np.zeros((n, 2), np.int32)}
    if not len(needed):
        return out
    dev = device()
    model = background_model(StreamReader(folder, "camera"), needed, dev)
    log(f"    background model from {model['frames_used']} frames, sky = {float(model['sky'].float().mean()):.0%} "
        f"of the image")
    net = timm.create_model(CAMERA_MODEL, pretrained=True, num_classes=0).eval().to(dev)
    loader = torch.utils.data.DataLoader(_Frames(folder, needed), batch_size=batch_size, num_workers=workers)
    avg, mx, where = [], [], []
    previous = None                 # last frames of the previous batch, so neighbours exist at batch edges
    with torch.no_grad():
        for i, rgb in enumerate(loader):
            rgb = rgb.to(dev, non_blocking=True)
            keep = len(rgb)
            if previous is not None:
                rgb = torch.cat([previous, rgb])
            found = locate(rgb, model)[-keep:]
            previous = rgb[-NEIGHBOUR:]
            rgb = rgb[-keep:]
            crops = torch.stack([torch.from_numpy(crop_at(rgb[k].cpu().numpy(), int(x), int(y)))
                                 for k, (x, y, _) in enumerate(found.tolist())]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                fmap = net.forward_features(standardise(crops))
            avg.append(fmap.float().mean((2, 3)).cpu().numpy())
            mx.append(fmap.float().amax((2, 3)).cpu().numpy())
            where.append(found.cpu().numpy())
            if i % 100 == 0:
                log(f"    crops {i * batch_size}/{len(needed)} frames")
    avg, mx, where = np.concatenate(avg), np.concatenate(mx), np.concatenate(where)
    position = {int(f): k for k, f in enumerate(needed)}
    for w, (a, b) in enumerate(frame_ranges):
        if b <= a:
            continue
        rows = np.array([position[int(f)] for f in range(a, b)])
        best = rows[np.argsort(-where[rows, 2])[:BEST_FRAMES]]
        out["camera_crop_avg"][w] = avg[best].mean(0)
        out["camera_crop_max"][w] = mx[best].mean(0)
        out["camera_crop_peak"][w] = where[best[0], 2]
        out["camera_crop_frame"][w] = needed[best[0]]
        out["camera_crop_xy"][w] = where[best[0], :2]
    out["camera_crop_avg"] = out["camera_crop_avg"].astype(np.float16)
    out["camera_crop_max"] = out["camera_crop_max"].astype(np.float16)
    return out


def contact_sheet(folder, result, path, windows=48, size=128):
    """A grid of the raw crops of evenly spaced windows, with the peak written on each (for checking by eye)."""
    from PIL import Image, ImageDraw
    from .store import StreamReader
    reader = StreamReader(folder, "camera")
    have = np.flatnonzero(result["camera_crop_frame"] >= 0)
    if not len(have):
        return None
    pick = have[np.unique(np.linspace(0, len(have) - 1, min(windows, len(have))).astype(int))]
    cols = 8
    sheet = Image.new("RGB", (cols * size, size * ((len(pick) + cols - 1) // cols)), "white")
    for k, w in enumerate(pick):
        rgb = decode(reader.payload(int(result["camera_crop_frame"][w])))
        x, y = result["camera_crop_xy"][w]
        tile = Image.fromarray(crop_at(rgb, int(x), int(y), size))
        ImageDraw.Draw(tile).text((3, 3), f"{result['camera_crop_peak'][w]:.0f}", fill=(255, 0, 0))
        sheet.paste(tile, (size * (k % cols), size * (k // cols)))
    sheet.save(path)
    return Path(path).name


def recording_crops(name, folder, step2_dir, out_dir, log=print):
    """Crop features of one recording, saved as <out_dir>/<name>.crop.npz, plus a contact sheet."""
    cpu = np.load(Path(step2_dir) / f"{name}.npz")
    result = crop_embeddings(folder, cpu["camera_frames"], log=log)
    result["window_id"] = cpu["window_id"]
    np.savez_compressed(Path(out_dir) / f"{name}.crop.npz", **result)
    sheet = contact_sheet(folder, result, Path(out_dir) / f"{name}_crops.jpg")
    peaks = result["camera_crop_peak"][result["camera_crop_frame"] >= 0]
    return {"windows": int(len(peaks)), "sheet": sheet,
            "peak_q10_50_90": [round(float(q), 1) for q in np.percentile(peaks, [10, 50, 90])] if len(peaks) else None}
