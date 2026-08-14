import json


def build_scan_config_dict(
    *,
    frames,
    lines,
    pixels,
    sequences,
    accumulations,
    max_detector,
    bidirectional,
    bidirectional_phase_shift,
    tcspc_bins=None,
    factor="1+0j",
    ptu_filename=None,
) -> dict:
    """Assemble the serialisable scan-config dict from primitive values.
    Readers treat every key as optional.
    """
    cfg: dict = {
        "scan": {
            "frames": int(frames),
            "lines": int(lines),
            "pixels": int(pixels),
            "sequences": int(sequences),
            "accumulations": [int(a) for a in accumulations],
            "max_detector": int(max_detector),
            "bidirectional": bool(bidirectional),
            "bidirectional_phase_shift": float(
                bidirectional_phase_shift if bidirectional else 0
            ),
        },
        "calibration": {
            "factor": str(factor),
        },
    }
    if tcspc_bins is not None:
        cfg["scan"]["tcspc_bins"] = int(tcspc_bins)
    if ptu_filename:
        cfg["ptu_filename"] = str(ptu_filename)
    return cfg


def save_config(path, cfg: dict) -> None:
    """Write *cfg* to *path* as indented JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_config(path) -> dict:
    """Read and return the JSON config at *path*."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
