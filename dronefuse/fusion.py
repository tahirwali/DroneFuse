"""
Fusion models for Step 3, and the metrics used to compare them.

Each 1 s window becomes eight sensor tokens: the camera (ConvNeXt avg + max pooling), the two
Livox LiDARs, the radar, and four microphone nodes (AST embedding + log-mel stats). Every token
also carries log(1 + count) of what the sensor delivered, so a model can tell a weak reading
from a strong one.

A sensor counts as present when it delivered at least `threshold` frames or points. We pick the
point threshold on the validation split, never on test.

The methods we compare are listed in METHODS. DroneFuse is an attention model that simply skips
the missing sensors. It is trained by "transplanting" the co-missing patterns seen in real
training windows, rather than dropping sensors at random as LMMUAVC does. See
docs/step3_fusion.md for the full description.
"""
import copy
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

from .recordings import BY_NAME, DRONE_TYPES
from .windows import protocol_sets

AUDIO_NODES = ("audio1", "audio2", "audio3", "audio4")
SENSORS = ("camera", "avia", "mid360", "radar") + AUDIO_NODES
GROUPS = {"camera": ("camera",), "audio": AUDIO_NODES, "avia": ("avia",), "mid360": ("mid360",),
          "radar": ("radar",)}
GROUP_NAMES = tuple(GROUPS)
COUNT_COLUMN = {"camera": "camera_frames", "avia": "livox_avia_points", "mid360": "livox_mid360_fg_points",
                "radar": "radar_fg_points", **{a: f"{a}_frames" for a in AUDIO_NODES}}
POINT_SENSORS = ("avia", "mid360", "radar")
THRESHOLD_CANDIDATES = (1, 5, 25)
FUSED = ("concat", "lmmuavc", "dronefuse", "dronefuse_iid", "dronefuse_mix")
METHODS = tuple(f"single_{g}" for g in GROUP_NAMES) + ("late",) + FUSED


# Data
@dataclass
class Data:
    ids: np.ndarray          # window ids
    recording: np.ndarray
    version: np.ndarray
    drone_type: np.ndarray
    split: np.ndarray
    x: dict                  # sensor -> (n, d) float32, raw (not standardised)
    counts: np.ndarray       # (n, len(SENSORS)) int64
    skipped: dict            # recording -> reason

    def __len__(self):
        return len(self.ids)


def _cpu_block(cpu, key, n, dim):
    return cpu[key].astype(np.float32).reshape(n, -1) if key in cpu.files else np.zeros((n, dim), np.float32)


def load_step2(step2_dir, windows):
    """All covered windows of every recording that has both Step 2 files."""
    step2_dir = Path(step2_dir)
    row_of = {w["window_id"]: w for w in windows}
    parts, skipped = [], {}
    for name in BY_NAME:
        cpu_path, deep_path = step2_dir / f"{name}.npz", step2_dir / f"{name}.deep.npz"
        if not cpu_path.exists():
            skipped[name] = "no Step 2 CPU file"
            continue
        if not deep_path.exists():
            skipped[name] = "no Step 2 GPU file"
            continue
        cpu, deep = np.load(cpu_path), np.load(deep_path)
        ids = np.array([str(i) for i in cpu["window_id"]])
        if not np.array_equal(ids, np.array([str(i) for i in deep["window_id"]])):
            skipped[name] = "CPU and GPU windows differ"
            continue
        n = len(ids)
        names = [str(s) for s in cpu["availability_names"]]
        avail = cpu["availability"]
        counts = np.stack([avail[:, names.index(COUNT_COLUMN[s])] for s in SENSORS], 1).astype(np.int64)
        x = {}
        if "camera_avg" in deep.files:
            x["camera"] = np.concatenate([deep["camera_avg"], deep["camera_max"]], 1).astype(np.float32)
        else:
            x["camera"] = np.zeros((n, 1536), np.float32)
            counts[:, SENSORS.index("camera")] = 0
        x["avia"] = _cpu_block(cpu, "livox_avia_all", n, 39)
        x["mid360"] = _cpu_block(cpu, "livox_mid360_fg", n, 39)
        x["radar"] = _cpu_block(cpu, "radar_fg", n, 49)
        ast, stats = deep["audio_ast"].astype(np.float32), cpu["audio_stats"].astype(np.float32)
        for k, node in enumerate(AUDIO_NODES):
            x[node] = np.concatenate([ast[:, k], stats[:, k]], 1)
        keep = cpu["covered"].astype(bool)
        rows = [row_of[i] for i in ids]
        parts.append({"ids": ids[keep], "x": {s: v[keep] for s, v in x.items()}, "counts": counts[keep],
                      "recording": np.array([r["recording"] for r in rows])[keep],
                      "version": np.array([r["version"] for r in rows])[keep],
                      "drone_type": np.array([r["drone_type"] for r in rows])[keep],
                      "split": np.array([r["split"] for r in rows])[keep]})
    if not parts:
        raise FileNotFoundError(f"no complete Step 2 recording in {step2_dir}")
    cat = lambda key: np.concatenate([p[key] for p in parts])
    x = {s: np.concatenate([p["x"][s] for p in parts]) for s in SENSORS}
    return Data(cat("ids"), cat("recording"), cat("version"), cat("drone_type"), cat("split"), x,
                np.concatenate([p["counts"] for p in parts]), skipped)


def presence(counts, point_threshold):
    """(n, n_sensors) bool: camera/audio need >= 1 frame, point sensors >= point_threshold points."""
    need = np.array([point_threshold if s in POINT_SENSORS else 1 for s in SENSORS])
    return counts >= need


def group_presence(mask):
    """(n, n_groups) bool: a group is present when any of its sensors is."""
    return np.stack([mask[:, [SENSORS.index(s) for s in GROUPS[g]]].any(1) for g in GROUP_NAMES], 1)


@dataclass
class Task:
    """One protocol: indices into Data, labels (-1 = unknown type) and the known class names."""
    protocol: str
    known: list
    unknown: list
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    y: np.ndarray


def make_task(data, windows, protocol):
    in_data = set(data.ids)
    sets = protocol_sets([w for w in windows if w["window_id"] in in_data], protocol)
    pos = {i: k for k, i in enumerate(data.ids)}
    idx = {s: np.array(sorted(pos[w["window_id"]] for w in sets[s]), np.int64) for s in ("train", "val", "test")}
    known = list(sets["known"])
    y = np.array([known.index(t) if t in known else -1 for t in data.drone_type], np.int64)
    if (y[idx["train"]] < 0).any() or (y[idx["val"]] < 0).any():
        raise AssertionError(f"{protocol}: an unknown type reached train or validation")
    return Task(protocol, known, list(sets["unknown"]), idx["train"], idx["val"], idx["test"], y)


def prepare_inputs(data, task, mask):
    """Standardised inputs (statistics from present TRAIN windows only), count channel appended,
    zero where the sensor is missing. Returns {sensor: (n, d + 1) float32}."""
    out = {}
    for j, s in enumerate(SENSORS):
        x = data.x[s]
        fit = task.train[mask[task.train, j]]
        if len(fit) < 2:
            fit = task.train
        mu, sd = x[fit].mean(0), x[fit].std(0) + 1e-6
        z = np.clip((x - mu) / sd, -10, 10)
        c = np.log1p(data.counts[:, j].astype(np.float32))
        c = c / max(float(c[task.train].max()), 1e-6)
        z = np.concatenate([z, c[:, None]], 1).astype(np.float32)
        z[~mask[:, j]] = 0.0
        out[s] = z
    return out


# Models
def _torch():
    import torch
    return torch


def build_model(method, dims, n_classes, d=128):
    torch = _torch()
    nn = torch.nn

    def mlp(din):
        return nn.Sequential(nn.LayerNorm(din), nn.Linear(din, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, d))

    class Tokens(nn.Module):
        def __init__(self, sensors):
            super().__init__()
            self.sensors = list(sensors)
            keys = sorted({"audio" if s in AUDIO_NODES else s for s in self.sensors})
            self.enc = nn.ModuleDict({k: mlp(dims[AUDIO_NODES[0] if k == "audio" else k]) for k in keys})
            self.emb = nn.Parameter(torch.randn(len(self.sensors), d) * 0.02)

        def forward(self, xs):
            t = [self.enc["audio" if s in AUDIO_NODES else s](xs[s]) for s in self.sensors]
            return torch.stack(t, 1) + self.emb

    class Single(nn.Module):
        def __init__(self, group):
            super().__init__()
            self.cols = [SENSORS.index(s) for s in GROUPS[group]]
            self.tokens = Tokens(GROUPS[group])
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

        def forward(self, xs, mask):
            t = self.tokens(xs)
            m = mask[:, self.cols].float().unsqueeze(-1)
            return self.head((t * m).sum(1) / m.sum(1).clamp(min=1.0))

    class Concat(nn.Module):
        def __init__(self):
            super().__init__()
            din = sum(dims[s] for s in SENSORS)
            self.net = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, 512), nn.GELU(), nn.Dropout(0.3),
                                     nn.Linear(512, n_classes))

        def forward(self, xs, mask):
            return self.net(torch.cat([xs[s] * mask[:, j:j + 1].float() for j, s in enumerate(SENSORS)], 1))

    class Slot(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = Tokens(SENSORS)
            self.missing = nn.Parameter(torch.randn(len(SENSORS), d) * 0.02)
            self.head = nn.Sequential(nn.LayerNorm(len(SENSORS) * d), nn.Linear(len(SENSORS) * d, 256), nn.GELU(),
                                      nn.Dropout(0.3), nn.Linear(256, n_classes))

        def forward(self, xs, mask):
            t = torch.where(mask.unsqueeze(-1), self.tokens(xs), self.missing.unsqueeze(0))
            return self.head(t.flatten(1))

    class Attention(nn.Module):
        def __init__(self, layers=2, heads=4):
            super().__init__()
            self.tokens = Tokens(SENSORS)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

        def forward(self, xs, mask):
            t = self.tokens(xs)
            t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
            pad = torch.cat([torch.zeros(len(t), 1, dtype=torch.bool, device=t.device), ~mask], 1)
            return self.head(self.encoder(t, src_key_padding_mask=pad)[:, 0])

    if method.startswith("single_"):
        return Single(method[len("single_"):])
    if method == "concat":
        return Concat()
    if method == "lmmuavc":
        return Slot()
    if method.startswith("dronefuse"):
        return Attention()
    raise ValueError(method)


# Training
AUGMENT = {"single": None, "concat": None, "lmmuavc": ("iid", 0.3), "dronefuse": ("pattern", 0.5),
           "dronefuse_iid": ("iid", 0.3), "dronefuse_mix": ("mix", 0.5)}


def augment_mask(mask, kind, p, pool, generator):
    """Remove sensors from a batch of presence masks; a window always keeps at least one sensor."""
    torch = _torch()
    new = mask.clone()
    if kind in ("pattern", "mix"):
        pick = torch.rand(len(mask), generator=generator, device=mask.device) < p
        other = pool[torch.randint(len(pool), (len(mask),), generator=generator, device=mask.device)]
        new = torch.where(pick.unsqueeze(1), new & other, new)
    if kind in ("iid", "mix"):
        q = p if kind == "iid" else 0.15
        new = new & ~(torch.rand(mask.shape, generator=generator, device=mask.device) < q)
    empty = ~new.any(1)
    new[empty] = mask[empty]
    return new


@dataclass
class TrainSettings:
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-2


def train(method, inputs, mask, task, seed, settings, device):
    """Returns the model with the best validation accuracy, and its training history."""
    torch = _torch()
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    n_classes = len(task.known)
    dims = {s: v.shape[1] for s, v in inputs.items()}
    model = build_model(method, dims, n_classes).to(device)
    xs = {s: torch.from_numpy(v).to(device) for s, v in inputs.items()}
    m_all = torch.from_numpy(mask).to(device)
    y_all = torch.from_numpy(task.y).to(device)
    tr = task.train
    if method.startswith("single_"):
        cols = [SENSORS.index(s) for s in GROUPS[method[len("single_"):]]]
        tr = tr[mask[tr][:, cols].any(1)]               # a single-sensor model learns only where it sees data
    if len(tr) == 0 or len(np.unique(task.y[tr])) < 2:
        return None, {"note": "no training windows with this sensor"}
    counts = np.bincount(task.y[tr], minlength=n_classes).astype(np.float32)
    weight = torch.from_numpy(np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
                              .astype(np.float32)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    steps = settings.epochs * math.ceil(len(tr) / settings.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=settings.lr, total_steps=steps, pct_start=0.1)
    aug = AUGMENT["single" if method.startswith("single_") else method]
    pool = m_all[torch.from_numpy(task.train).to(device)]
    tr_t = torch.from_numpy(tr).to(device)
    best, best_acc, history = None, -1.0, []
    for epoch in range(settings.epochs):
        model.train()
        order = tr_t[torch.randperm(len(tr_t), generator=generator, device=device)]
        for b in range(0, len(order), settings.batch_size):
            i = order[b:b + settings.batch_size]
            m = m_all[i]
            if aug is not None:
                m = augment_mask(m, aug[0], aug[1], pool, generator)
            loss = loss_fn(model({s: v[i] for s, v in xs.items()}, m), y_all[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        logits = predict(model, inputs, mask, task.val, device)
        acc = float((logits.argmax(1) == task.y[task.val]).mean())
        history.append(round(acc, 4))
        if acc > best_acc:
            best_acc, best = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model, {"val_accuracy_per_epoch": history, "best_val_accuracy": best_acc,
                   "best_epoch": int(np.argmax(history)) + 1, "train_windows": int(len(tr))}


def predict(model, inputs, mask, index, device, batch=2048):
    """Logits (len(index), n_classes) as float32 numpy."""
    torch = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(index), batch):
            i = index[b:b + batch]
            xs = {s: torch.from_numpy(v[i]).to(device) for s, v in inputs.items()}
            out.append(model(xs, torch.from_numpy(mask[i]).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def single_logits(logits, mask, group, index):
    """A single-group model says nothing (uniform: all-zero logits) where its group is missing."""
    cols = [SENSORS.index(s) for s in GROUPS[group]]
    out = logits.copy()
    out[~mask[index][:, cols].any(1)] = 0.0
    return out


def late_fusion(single, gmask):
    """single: {group: logits (n, K)}; gmask (n, n_groups). Log of the mean probability of present groups."""
    probs, weight = 0.0, 0.0
    for j, g in enumerate(GROUP_NAMES):
        if g not in single:
            continue
        p = softmax(single[g])
        w = gmask[:, j:j + 1].astype(np.float64)
        probs, weight = probs + p * w, weight + w
    k = next(iter(single.values())).shape[1]
    mean = np.where(weight > 0, probs / np.maximum(weight, 1e-12), 1.0 / k)
    return np.log(np.clip(mean, 1e-12, None)).astype(np.float32)


# Scores and metrics
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def energy(z):
    m = z.max(1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]


def scores(logits, n_groups, val_logits, val_groups, min_bin=20):
    """{"msp", "energy", "avail_energy"}: higher means 'known'. Availability normalisation uses
    validation windows only, binned by the number of sensor groups present."""
    e, ev = energy(logits), energy(val_logits)
    norm = np.empty_like(e)
    mu_all, sd_all = ev.mean(), ev.std() + 1e-6
    for b in np.unique(n_groups):
        sel = val_groups == b
        mu, sd = (ev[sel].mean(), ev[sel].std() + 1e-6) if sel.sum() >= min_bin else (mu_all, sd_all)
        norm[n_groups == b] = (e[n_groups == b] - mu) / sd
    return {"msp": softmax(logits).max(1), "energy": e, "avail_energy": norm}


def auroc(known_scores, unknown_scores):
    """Probability that a known window scores higher than an unknown one (ties count half)."""
    if not len(known_scores) or not len(unknown_scores):
        return None
    s = np.concatenate([known_scores, unknown_scores])
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                                  # average ranks of ties
        tie = s == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    n1 = len(known_scores)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(unknown_scores)))


def fpr_at_95(known_scores, unknown_scores):
    if not len(known_scores) or not len(unknown_scores):
        return None
    threshold = np.percentile(known_scores, 5)
    return float((unknown_scores >= threshold).mean())


def macro_f1(y, pred, k):
    f1 = []
    for c in range(k):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        if tp + fp + fn:
            f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1)) if f1 else None


def _predict_class(logits):
    pred = logits.argmax(1)
    pred[logits.max(1) == logits.min(1)] = -1
    return pred


def evaluate(test_logits, y_test, test_groups, val_logits, y_val, val_groups):
    """All metrics of one run. y = -1 for unknown types. *_groups = number of groups present."""
    k = val_logits.shape[1]
    known = y_test >= 0
    pred = _predict_class(test_logits)                       # no opinion (sensor missing) counts as wrong
    out = {"known_accuracy": float((pred[known] == y_test[known]).mean()) if known.any() else None,
           "known_macro_f1": macro_f1(y_test[known], pred[known], k) if known.any() else None,
           "val_accuracy": float((_predict_class(val_logits) == y_val).mean()),
           "test_known": int(known.sum()), "test_unknown": int((~known).sum())}
    s_test = scores(test_logits, test_groups, val_logits, val_groups)
    s_val = scores(val_logits, val_groups, val_logits, val_groups)
    for name in s_test:
        st = s_test[name]
        threshold = float(np.percentile(s_val[name], 5))           # keeps 95 % of validation windows
        accept = st >= threshold
        correct = np.where(known, accept & (pred == y_test), ~accept)
        out[name] = {"auroc": auroc(st[known], st[~known]), "fpr_at_95_tpr": fpr_at_95(st[known], st[~known]),
                     "open_accuracy": float(correct.mean()),
                     "known_accepted": float(accept[known].mean()) if known.any() else None,
                     "unknown_rejected": float((~accept[~known]).mean()) if (~known).any() else None}
    by_groups = {}
    for b in range(1, len(GROUP_NAMES) + 1):
        sel = known & (test_groups == b)
        if sel.sum():
            by_groups[str(b)] = {"windows": int(sel.sum()), "accuracy": float((pred[sel] == y_test[sel]).mean())}
    out["known_accuracy_by_groups_present"] = by_groups
    return out


def all_subsets():
    """The 31 non-empty subsets of the sensor groups, as tuples of group names."""
    return [c for r in range(1, len(GROUP_NAMES) + 1) for c in combinations(GROUP_NAMES, r)]


def subset_mask(mask, groups):
    keep = np.array([any(s in GROUPS[g] for g in groups) for s in SENSORS])
    return mask & keep[None, :]


def perturbed_masks(mask, index, seed):
    """Test-time removal beyond the natural one: independent (p = 0.3) and co-missing patterns
    transplanted from other test windows (p = 0.5). Every window keeps at least one sensor."""
    rng = np.random.default_rng(1000 + seed)
    m = mask[index]
    iid = m & ~(rng.random(m.shape) < 0.3)
    pattern = m.copy()
    pick = rng.random(len(m)) < 0.5
    pattern[pick] &= m[rng.integers(len(m), size=int(pick.sum()))]
    for new in (iid, pattern):
        empty = ~new.any(1)
        new[empty] = m[empty]
    return {"iid_drop": iid, "pattern_drop": pattern}
