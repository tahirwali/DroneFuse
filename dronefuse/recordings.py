"""
The 12 MMAUD sensor recordings, their drone labels, and the Step 1 extraction groups.

Class = drone type. The Mavic 3 (V1) and Mavic 3 Enterprise (V2/V3, `m3e`) share an airframe
and are one class, "Mavic3"; the finer model name is kept in `model` for analysis.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Recording:
    name: str            # e.g. "V1_Avata", "V2_m30s1"
    version: str         # "V1" (rooftop) or "V2_V3" (carpark)
    drone_type: str      # the class label
    model: str           # finer model name
    sensor_bag: str      # path on the Hub (byte parts are found from the inventory)
    truth_bag: str


def _v1(name, drone_type, model):
    return Recording(f"V1_{name}", "V1", drone_type, model, f"V1/rosbag/{name}.bag", f"V1/ground_truth/{name}.bag")


def _v2(name, drone_type, model):
    return Recording(f"V2_{name}", "V2_V3", drone_type, model, f"V2_V3/rosbag/{name}.bag",
                     f"V2_V3/ground_truth/{name}.bag")


RECORDINGS = [
    _v1("Avata", "Avata", "DJI Avata"),
    _v1("M300", "M300", "DJI Matrice 300 RTK"),
    _v1("Mavic2", "Mavic2", "DJI Mavic 2"),
    _v1("Mavic3", "Mavic3", "DJI Mavic 3"),
    _v1("Pham4", "Phantom4", "DJI Phantom 4"),
    _v2("m300s1", "M300", "DJI Matrice 300 RTK"),
    _v2("m300s2", "M300", "DJI Matrice 300 RTK"),
    _v2("m30s1", "M30", "DJI Matrice 30"),
    _v2("m30s2", "M30", "DJI Matrice 30"),
    _v2("m3e", "Mavic3", "DJI Mavic 3 Enterprise"),
    _v2("p4s1", "Phantom4", "DJI Phantom 4"),
    _v2("p4s2", "Phantom4", "DJI Phantom 4"),
]
BY_NAME = {r.name: r for r in RECORDINGS}
DRONE_TYPES = sorted({r.drone_type for r in RECORDINGS})

# One Kaggle CPU session per group; each saves well under the 20 GB output limit.
GROUPS = {
    "v1": ["V1_Avata", "V1_M300", "V1_Mavic2", "V1_Mavic3", "V1_Pham4"],                  # 77.9 GB of bags
    "v2a": ["V2_m300s1", "V2_m300s2", "V2_m30s1", "V2_m30s2"],                          # 78.6 GB
    "v2b": ["V2_m3e", "V2_p4s1", "V2_p4s2"],                                            # 115.0 GB
}
