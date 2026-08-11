"""Read/write scan-configuration files (JSON).

Schema::

    {
      "ptu_filename": "scan.ptu",     # optional — recorded for reference only;
                                      #   Load Config never uses it
      "scan": {
        "frames", "lines", "pixels", "sequences",
        "accumulations": [1, 3, ...], # one int per sequence
        "max_detector",
        "tcspc_bins",                 # optional — omitted by writers that
                                      #   have no such field (e.g. Batch)
        "bidirectional", "bidirectional_phase_shift"
      },
      "calibration": {"f_rep_mhz", "factor": "1+0j"}
    }
"""

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
    f_rep_mhz,
    tcspc_bins=None,
    factor="1+0j",
    ptu_filename=None,
) -> dict:
    """Assemble the serialisable scan-config dict from primitive values.

    ``tcspc_bins`` is omitted from the result when None, for callers that have
    no such field; readers treat every key as optional.
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
            "f_rep_mhz": float(f_rep_mhz),
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
