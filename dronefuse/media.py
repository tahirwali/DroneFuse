"""Describe downloaded sample files (PNG, NPY, WAV) so we know their real format before extraction."""
from pathlib import Path


def inspect_png(path):
    from PIL import Image
    with Image.open(path) as image:
        return {"kind": "image", "width": image.width, "height": image.height, "mode": image.mode,
                "file_mb": round(Path(path).stat().st_size / 1e6, 2)}


def inspect_npy(path):
    import numpy as np
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    out = {"kind": "array", "shape": list(array.shape), "dtype": str(array.dtype),
           "file_mb": round(Path(path).stat().st_size / 1e6, 3)}
    if array.size and array.dtype.kind in "iuf":
        flat = np.asarray(array).reshape(-1)[:1_000_000].astype("float64")
        out.update(min=float(flat.min()), max=float(flat.max()), mean=float(flat.mean()))
        if array.ndim == 2 and array.shape[1] <= 8:
            out["first_row"] = [float(v) for v in np.asarray(array[0])]
    elif array.ndim == 1 and array.size <= 8:
        out["values"] = np.asarray(array).tolist()
    return out


def inspect_audio(path):
    try:
        import soundfile
        info = soundfile.info(str(path))
        return {"kind": "audio", "sample_rate": info.samplerate, "channels": info.channels,
                "frames": info.frames, "seconds": round(info.duration, 1), "subtype": info.subtype,
                "file_mb": round(Path(path).stat().st_size / 1e6, 1)}
    except Exception:
        from scipy.io import wavfile
        rate, data = wavfile.read(str(path), mmap=True)
        channels = 1 if data.ndim == 1 else data.shape[1]
        return {"kind": "audio", "sample_rate": rate, "channels": channels, "frames": int(data.shape[0]),
                "seconds": round(data.shape[0] / rate, 1), "subtype": str(data.dtype),
                "file_mb": round(Path(path).stat().st_size / 1e6, 1)}


def inspect_file(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".png" or suffix in (".jpg", ".jpeg"):
        return inspect_png(path)
    if suffix == ".npy":
        return inspect_npy(path)
    if suffix in (".wav", ".mp3"):
        return inspect_audio(path)
    return {"kind": "other", "file_mb": round(Path(path).stat().st_size / 1e6, 3)}


def save_thumbnail(path, out_path, width=480):
    from PIL import Image
    with Image.open(path) as image:
        image.thumbnail((width, width))
        image.save(out_path)
