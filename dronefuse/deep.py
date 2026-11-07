"""
Step 2, GPU part: frozen pretrained embeddings of the camera frames and the audio windows.

camera  ConvNeXt-T (ImageNet-1k, timm `convnext_tiny.fb_in1k`; the same backbone LMMUAVC uses) on the
        native 1280x960 left image. The last feature map (40x30 cells of 32x32 pixels) is pooled two
        ways: average (what a whole-frame classifier sees) and max (keeps a small, strong local
        response such as a few-pixel drone). Per window: mean over its frames of both.
audio   AST fine-tuned on AudioSet (`MIT/ast-finetuned-audioset-10-10-0.4593`), 1 s windows at 16 kHz
        (padded by its feature extractor, as for short AudioSet clips). Per window and node: the pooled
        embedding (768).
Nothing is fine-tuned. torch, timm and transformers are imported only here (Kaggle has all three).
"""
import io

import numpy as np

CAMERA_MODEL = "convnext_tiny.fb_in1k"
AUDIO_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class _Frames:
    """torch Dataset of JPEG frames of one recording, decoded in DataLoader workers."""

    def __init__(self, folder, indices, mean, std):
        self.folder, self.indices, self.mean, self.std = folder, indices, mean, std
        self.reader = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        import torch
        from PIL import Image
        from .store import StreamReader
        if self.reader is None:
            self.reader = StreamReader(self.folder, "camera")
        image = np.asarray(Image.open(io.BytesIO(self.reader.payload(int(self.indices[k])))).convert("RGB"))
        x = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(x.transpose(2, 0, 1).copy())


def camera_embeddings(folder, frame_ranges, batch_size=8, workers=4, log=print):
    """
    Embed every camera frame used by at least one window, then average per window.
    Returns {"camera_avg": (n, 768), "camera_max": (n, 768)} as float16 (zeros for a window with no frames).
    """
    import timm
    import torch
    needed = np.unique(np.concatenate([np.arange(a, b) for a, b in frame_ranges] or [np.zeros(0, np.int64)]))
    n = len(frame_ranges)
    if not len(needed):
        return {"camera_avg": np.zeros((n, 768), np.float16), "camera_max": np.zeros((n, 768), np.float16)}
    dev = device()
    model = timm.create_model(CAMERA_MODEL, pretrained=True, num_classes=0).eval().to(dev)
    config = model.pretrained_cfg
    mean, std = np.array(config["mean"], np.float32), np.array(config["std"], np.float32)
    loader = torch.utils.data.DataLoader(_Frames(folder, needed, mean, std), batch_size=batch_size,
                                         num_workers=workers, pin_memory=dev == "cuda")
    avg, mx = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for i, batch in enumerate(loader):
            fmap = model.forward_features(batch.to(dev, non_blocking=True))     # (B, 768, 30, 40)
            avg.append(fmap.float().mean((2, 3)).cpu().numpy())
            mx.append(fmap.float().amax((2, 3)).cpu().numpy())
            if i % 200 == 0:
                log(f"    camera {i * batch_size}/{len(needed)} frames")
    avg, mx = np.concatenate(avg), np.concatenate(mx)
    position = {int(f): k for k, f in enumerate(needed)}
    out_avg = np.zeros((n, avg.shape[1]), np.float32)
    out_max = np.zeros_like(out_avg)
    for w, (a, b) in enumerate(frame_ranges):
        if b > a:
            rows = [position[int(f)] for f in range(a, b)]
            out_avg[w], out_max[w] = avg[rows].mean(0), mx[rows].mean(0)
    return {"camera_avg": out_avg.astype(np.float16), "camera_max": out_max.astype(np.float16)}


def audio_embeddings(signals_per_node, present, batch_size=32, log=print):
    """
    signals_per_node: {node: (n, 16000) float32}; present: (n, n_nodes) bool (window has audio).
    Returns {"audio_ast": (n, n_nodes, 768) float16}; zeros where the node has no audio.
    """
    import torch
    from transformers import ASTFeatureExtractor, ASTModel
    dev = device()
    extractor = ASTFeatureExtractor.from_pretrained(AUDIO_MODEL)
    model = ASTModel.from_pretrained(AUDIO_MODEL).eval().to(dev)
    nodes = list(signals_per_node)
    n = len(next(iter(signals_per_node.values())))
    out = np.zeros((n, len(nodes), model.config.hidden_size), np.float16)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for k, node in enumerate(nodes):
            rows = np.flatnonzero(present[:, k])
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                inputs = extractor(list(signals_per_node[node][batch]), sampling_rate=16000, return_tensors="pt")
                pooled = model(input_values=inputs["input_values"].to(dev)).pooler_output
                out[batch, k] = pooled.float().cpu().numpy()
            log(f"    {node}: {len(rows)} windows embedded")
    return {"audio_ast": out}
