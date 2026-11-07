"""
List every file of the MMAUD dataset on the Hugging Face Hub with its size, and total the
sizes per folder (one folder = one sequence/modality, e.g. V1/filesystem_format/Mavic2/image).
"""
import csv
import gzip
from collections import defaultdict


def list_files(repo_id, repo_type="dataset"):
    """Return [(path, size_in_bytes)] for every file in the repository."""
    from huggingface_hub import HfApi
    files = []
    for entry in HfApi().list_repo_tree(repo_id, repo_type=repo_type, recursive=True):
        size = getattr(entry, "size", None)
        if size is not None and not hasattr(entry, "tree_id"):   # folders have tree_id and no size
            files.append((entry.path, int(size)))
    return sorted(files)


def folder_totals(files):
    """[(folder, number of files, GB)] grouped by each file's parent folder, largest first."""
    totals = defaultdict(lambda: [0, 0])
    for path, size in files:
        folder = path.rsplit("/", 1)[0] if "/" in path else "."
        totals[folder][0] += 1
        totals[folder][1] += size
    rows = [(folder, n, round(size / 1e9, 3)) for folder, (n, size) in totals.items()]
    return sorted(rows, key=lambda r: -r[2])


def files_in(files, folder):
    """Files directly inside `folder`, sorted by name (MMAUD names are timestamps, so by time)."""
    prefix = folder.rstrip("/") + "/"
    return [(p, s) for p, s in files if p.startswith(prefix) and "/" not in p[len(prefix):]]


def save_inventory(files, path):
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "bytes"])
        writer.writerows(files)


def load_inventory(path):
    with gzip.open(path, "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [(p, int(s)) for p, s in reader]
