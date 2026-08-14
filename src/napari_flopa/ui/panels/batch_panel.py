"""
Batch analysis panel.

Reconstructs every .ptu file in a directory with a shared ScanConfig +
calibration factor and writes any combination of:
  • FLIM RGB / Intensity / Lifetime  TIFF images
  • Phasor CSV tables  (per-object intensity-weighted means, or per-pixel)
  • Decay CSV tables   (time_ns column + one column per curve combination)

Each export type has independent aggregation passes: add multiple rows to
run e.g. "no aggregation" AND "sum-frames + sum-detectors" in one batch.

Labels (optional): integer TIFFs in a separate folder matched by PTU name
prefix (0 = background).  Multiple label files per PTU are merged into one
per-file table; 'labels_file' is a column in every row.

Output: written to  <ptu_dir>/batch_<timestamp>/
Processing: sequential in a QThread — UI stays responsive; Stop aborts
between files.

Scan config + calibration can be saved/loaded as JSON (stdlib ``json``).
"""

import csv
import datetime
import itertools
import traceback
from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, QRegularExpression, Qt, QThread, Signal, Slot
from qtpy.QtGui import (
    QDoubleValidator,
    QIntValidator,
    QRegularExpressionValidator,
)
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_flopa.core.io.config import load_config, save_config
from napari_flopa.ui.state import FlopaState
from napari_flopa.ui.style import S, apply_style

# ── log colour helpers ────────────────────────────────────────────────────
_ERR = '<span style="color:#ff6060">{}</span>'
_OK = '<span style="color:#60cc60">{}</span>'
_INFO = '<span style="color:#aaaaaa">{}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# _AggRow  — one aggregation / processing configuration row
# ─────────────────────────────────────────────────────────────────────────────


class _AggRow(QWidget):
    """
    One aggregation pass: F / S / D checkboxes plus kind-specific extras.

    kind='images'  → Smooth checkbox + kernel spinbox
    kind='decay'   → Shift spinbox + Norm checkbox
    kind='phasor'  → no extras (mode is shared across all rows)

    Emits remove_requested(self) when × is clicked.
    """

    remove_requested = Signal(object)

    def __init__(
        self, kind: str = "phasor", removable: bool = True, parent=None
    ):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(6)

        self.frames_chk = QCheckBox("F")
        self.frames_chk.setToolTip("Sum all frames")
        self.seqs_chk = QCheckBox("S")
        self.seqs_chk.setToolTip("Sum all sequences")
        self.dets_chk = QCheckBox("D")
        self.dets_chk.setToolTip("Sum all detectors")
        for w in (self.frames_chk, self.seqs_chk, self.dets_chk):
            lay.addWidget(w)

        self.smooth_chk = self.smooth_spin = None
        self.shift_spin = self.norm_chk = None

        if kind == "images":
            lay.addWidget(_vsep())
            self.smooth_chk = QCheckBox("Smooth")
            self.smooth_spin = QSpinBox()
            self.smooth_spin.setRange(2, 50)
            self.smooth_spin.setValue(3)
            self.smooth_spin.setFixedWidth(42)
            self.smooth_spin.setEnabled(False)
            self.smooth_chk.toggled.connect(self.smooth_spin.setEnabled)
            lay.addWidget(self.smooth_chk)
            lay.addWidget(self.smooth_spin)

        elif kind == "decay":
            lay.addWidget(_vsep())
            lay.addWidget(QLabel("Shift:"))
            self.shift_spin = QSpinBox()
            self.shift_spin.setRange(-500, 500)
            self.shift_spin.setFixedWidth(54)
            lay.addWidget(self.shift_spin)
            self.norm_chk = QCheckBox("Norm")
            lay.addWidget(self.norm_chk)

        lay.addStretch()
        if removable:
            rm = QPushButton("×")
            rm.setFixedSize(18, 18)
            rm.setStyleSheet(
                "QPushButton { color: #888; border: none; font-size: 11px; }"
            )
            rm.setToolTip("Remove this pass")
            rm.clicked.connect(lambda: self.remove_requested.emit(self))
            lay.addWidget(rm)

    def to_dict(self) -> dict:
        d = dict(
            agg_frames=self.frames_chk.isChecked(),
            agg_seqs=self.seqs_chk.isChecked(),
            agg_dets=self.dets_chk.isChecked(),
        )
        if self.smooth_chk is not None:
            d["smooth"] = self.smooth_chk.isChecked()
            d["smooth_kernel"] = self.smooth_spin.value()
        if self.shift_spin is not None:
            d["shift"] = self.shift_spin.value()
            d["norm"] = self.norm_chk.isChecked()
        return d


def _vsep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setStyleSheet(S.SEPARATOR)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Export sections  (checkable QGroupBox + list of _AggRow)
# ─────────────────────────────────────────────────────────────────────────────


class _ExportSection(QGroupBox):
    """
    Checkable QGroupBox for one export type.
    Contains type-specific controls followed by a list of _AggRow passes.
    """

    def __init__(self, title: str, kind: str, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(False)
        apply_style(self, S.GROUP_PRIMARY)
        self._kind = kind
        self._rows: list[_AggRow] = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 6)
        lay.setSpacing(3)

        self._add_type_widgets(lay)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(S.SEPARATOR)
        lay.addWidget(sep)

        hint = QLabel(
            "Aggregation passes — one export per row  (F=Frames S=Seq D=Det):"
        )
        hint.setStyleSheet(S.HINT)
        lay.addWidget(hint)

        self._rows_widget = QWidget()
        self._rows_lay = QVBoxLayout(self._rows_widget)
        self._rows_lay.setContentsMargins(0, 0, 0, 0)
        self._rows_lay.setSpacing(1)
        lay.addWidget(self._rows_widget)

        add_btn = QPushButton("+ Add pass")
        add_btn.setFixedWidth(86)
        add_btn.setStyleSheet("font-weight: normal; font-size: 10px;")
        add_btn.clicked.connect(lambda: self._add_row(removable=True))
        lay.addWidget(add_btn)

        self._add_row(removable=False)  # default first row (not removable)

    def _add_type_widgets(self, lay: QVBoxLayout):
        """Override in subclasses to insert type-specific widgets above agg rows."""

    def _add_row(self, *, removable: bool = True):
        row = _AggRow(kind=self._kind, removable=removable)
        row.remove_requested.connect(self._remove_row)
        self._rows.append(row)
        self._rows_lay.addWidget(row)

    def _remove_row(self, row: _AggRow):
        if len(self._rows) <= 1:
            return
        self._rows.remove(row)
        self._rows_lay.removeWidget(row)
        row.deleteLater()

    def get_configs(self) -> list[dict]:
        """Return list of aggregation config dicts (empty if section is disabled)."""
        if not self.isChecked():
            return []
        base = self._type_config()
        return [{**base, **row.to_dict()} for row in self._rows]

    def _type_config(self) -> dict:
        """Override to supply type-specific fields shared across all rows."""
        return {}


class _ImagesSection(_ExportSection):
    """Export TIFF images: FLIM RGB, Intensity, Lifetime."""

    def __init__(self, parent=None):
        super().__init__("Images (TIFF)", kind="images", parent=parent)

    def _add_type_widgets(self, lay):
        # Image type checkboxes
        row1 = QHBoxLayout()
        self._flim_chk = QCheckBox("FLIM RGB")
        self._int_chk = QCheckBox("Intensity")
        self._lt_chk = QCheckBox("Lifetime")
        self._flim_chk.setChecked(True)
        self._int_chk.setChecked(True)
        self._lt_chk.setChecked(True)
        for w in (self._flim_chk, self._int_chk, self._lt_chk):
            row1.addWidget(w)
            w.setStyleSheet("font-weight: normal;")
        row1.addStretch()
        lay.addLayout(row1)

        # FLIM RGB colormap + contrast
        row2 = QHBoxLayout()
        lbl2 = QLabel("Colormap:")
        lbl2.setStyleSheet(S.MUTED)
        row2.addWidget(lbl2)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["rainbow", "hsv", "viridis"])
        self._cmap_combo.setMaximumWidth(90)
        self._cmap_combo.setStyleSheet("font-weight: normal;")
        row2.addWidget(self._cmap_combo)
        row2.addSpacing(8)
        lbl3 = QLabel("LT range (ns):")
        lbl3.setStyleSheet(S.MUTED)
        row2.addWidget(lbl3)
        self._lt_lo = QLineEdit()
        self._lt_lo.setMaximumWidth(55)
        self._lt_hi = QLineEdit()
        self._lt_hi.setMaximumWidth(55)
        self._lt_lo.setPlaceholderText("auto")
        self._lt_hi.setPlaceholderText("auto")
        self._lt_lo.setValidator(QDoubleValidator(0.0, 9999.0, 4))
        self._lt_hi.setValidator(QDoubleValidator(0.0, 9999.0, 4))
        self._lt_lo.setToolTip(
            "Minimum lifetime (ns) for FLIM RGB LUT — leave empty for auto (data min)"
        )
        self._lt_hi.setToolTip(
            "Maximum lifetime (ns) for FLIM RGB LUT — leave empty for auto (data max)"
        )
        row2.addWidget(self._lt_lo)
        row2.addWidget(QLabel("–"))
        row2.addWidget(self._lt_hi)
        row2.addStretch()
        lay.addLayout(row2)
        self._flim_chk.toggled.connect(
            lambda on: (
                self._lt_lo.setEnabled(on),
                self._lt_hi.setEnabled(on),
                self._cmap_combo.setEnabled(on),
            )
        )

        # Export mode: aggregation passes OR single-frame pick
        row3 = QHBoxLayout()
        self._mode_grp = QButtonGroup(self)
        self._all_radio = QRadioButton("Aggregation passes")
        self._sf_radio = QRadioButton("Single frame")
        self._all_radio.setChecked(True)
        for r in (self._all_radio, self._sf_radio):
            self._mode_grp.addButton(r)
            r.setStyleSheet("font-weight: normal;")
            row3.addWidget(r)
        row3.addStretch()
        lay.addLayout(row3)

        # Single-frame selector (hidden until mode switched)
        self._sf_widget = QWidget()
        sf_lay = QHBoxLayout(self._sf_widget)
        sf_lay.setContentsMargins(0, 0, 0, 0)
        sf_lay.setSpacing(4)
        for txt in ("Frame:", "Seq:", "Det:"):
            sf_lay.addWidget(QLabel(txt))
            sp = QSpinBox()
            sp.setRange(0, 9999)
            sp.setFixedWidth(52)
            sp.setStyleSheet("font-weight: normal;")
            sf_lay.addWidget(sp)
            setattr(self, f"_sf_{txt[:-1].lower()}", sp)
        sf_lay.addStretch()
        self._sf_widget.setVisible(False)
        lay.addWidget(self._sf_widget)

        self._sf_radio.toggled.connect(self._on_mode_toggled)

    def _on_mode_toggled(self, single: bool):
        self._sf_widget.setVisible(single)
        # Disable/enable the aggregation passes section
        self._rows_widget.setEnabled(not single)

    def get_configs(self) -> list[dict]:
        if not self.isChecked():
            return []
        base = self._type_config()
        if base.get("single_frame"):
            return [base]
        return [{**base, **row.to_dict()} for row in self._rows]

    def _type_config(self):
        lo_txt = self._lt_lo.text().strip()
        hi_txt = self._lt_hi.text().strip()
        lt_ns_lo = float(lo_txt) if lo_txt else None
        lt_ns_hi = float(hi_txt) if hi_txt else None
        cfg = dict(
            export_flim=self._flim_chk.isChecked(),
            export_int=self._int_chk.isChecked(),
            export_lt=self._lt_chk.isChecked(),
            lt_ns_lo=lt_ns_lo,
            lt_ns_hi=lt_ns_hi,
            cmap=self._cmap_combo.currentText(),
            single_frame=self._sf_radio.isChecked(),
        )
        if self._sf_radio.isChecked():
            cfg.update(
                sf_frame=self._sf_frame.value(),
                sf_seq=self._sf_seq.value(),
                sf_det=self._sf_det.value(),
            )
        return cfg


class _PhasorSection(_ExportSection):
    """Export phasor table CSV: per object (intensity-weighted) or per pixel."""

    def __init__(self, parent=None):
        super().__init__("Phasor table (CSV)", kind="phasor", parent=parent)

    def _add_type_widgets(self, lay):
        row1 = QHBoxLayout()
        self._grp = QButtonGroup(self)
        self._obj_radio = QRadioButton("Per object")
        self._px_radio = QRadioButton("Per pixel")
        self._obj_radio.setChecked(True)
        for r in (self._obj_radio, self._px_radio):
            self._grp.addButton(r)
            r.setStyleSheet("font-weight: normal;")
            row1.addWidget(r)
        row1.addStretch()
        lay.addLayout(row1)
        self._warn = QLabel(
            "⚠  Per-pixel tables can have millions of rows per file."
        )
        self._warn.setStyleSheet(S.WARNING)
        self._warn.setVisible(False)
        lay.addWidget(self._warn)
        self._px_radio.toggled.connect(self._warn.setVisible)

        row2 = QHBoxLayout()
        self._smooth_chk = QCheckBox("Smooth G/S before extraction")
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(2, 50)
        self._smooth_spin.setValue(3)
        self._smooth_spin.setFixedWidth(42)
        self._smooth_spin.setEnabled(False)
        self._smooth_chk.setStyleSheet("font-weight: normal;")
        self._smooth_chk.toggled.connect(self._smooth_spin.setEnabled)
        row2.addWidget(self._smooth_chk)
        row2.addWidget(self._smooth_spin)
        row2.addStretch()
        lay.addLayout(row2)

    def _type_config(self):
        return dict(
            per_pixel=self._px_radio.isChecked(),
            smooth=self._smooth_chk.isChecked(),
            smooth_kernel=self._smooth_spin.value(),
        )


class _DecaySection(_ExportSection):
    """Export decay CSV: time_ns + one column per curve (always per-file)."""

    def __init__(self, parent=None):
        super().__init__(
            "Decay table (CSV, per file)", kind="decay", parent=parent
        )


# ─────────────────────────────────────────────────────────────────────────────
# _BatchWorker  — runs in QThread
# ─────────────────────────────────────────────────────────────────────────────


class _BatchWorker(QObject):
    """
    Sequential batch processor.

    Signals:
        progress(current, total) — file index update
        log(html)                — HTML-formatted log line
        finished(summary)        — plain-text summary
    """

    progress = Signal(int, int)
    log = Signal(str)
    finished = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self._p = params
        self._stop = False

    def stop(self):
        """Request abort after the current file finishes."""
        self._stop = True

    @Slot()
    def run(self):
        p = self._p
        ptu_dir = Path(p["ptu_dir"])
        lbl_dir = Path(p["lbl_dir"]) if p.get("lbl_dir") else None
        out_dir: Path = p["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        glob = "**/*.ptu" if p.get("recursive") else "*.ptu"
        ptu_files = sorted(ptu_dir.glob(glob))
        if not ptu_files:
            self.log.emit(_ERR.format("No .ptu files found."))
            self.finished.emit("No files processed.")
            return

        n_ok = n_err = 0
        phasor_merged: list[dict] = []

        for i, path in enumerate(ptu_files):
            if self._stop:
                self.log.emit(_INFO.format("Stopped by user."))
                break
            self.progress.emit(i, len(ptu_files))
            self.log.emit(
                _INFO.format(f"[{i+1}/{len(ptu_files)}]  {path.name}")
            )
            try:
                ph_rows = self._process_one(path, lbl_dir, out_dir, p)
                phasor_merged.extend(ph_rows)
                n_ok += 1
                self.log.emit(_OK.format("  ✓"))
            except Exception as exc:
                n_err += 1
                self.log.emit(_ERR.format(f"  ✗  {exc}"))
                traceback.print_exc()

        self.progress.emit(len(ptu_files), len(ptu_files))

        # Single merged phasor table
        if p.get("single_table") and phasor_merged and p.get("phasor_configs"):
            _write_csv(out_dir / "phasor_all.csv", phasor_merged)
            self.log.emit(
                _OK.format(
                    f"Merged phasor table: {out_dir / 'phasor_all.csv'}"
                )
            )

        summary = (
            f"Done: {n_ok} succeeded, {n_err} failed.  Output → {out_dir}"
        )
        self.log.emit((_OK if n_err == 0 else _ERR).format(summary))
        self.finished.emit(summary)

    # ── per-file dispatch ────────────────────────────────────────────────

    def _process_one(
        self, ptu_path: Path, lbl_dir, out_dir: Path, p: dict
    ) -> list[dict]:
        """
        Reconstruct one PTU file, apply calibration, run all export passes.

        Returns list of phasor row dicts (for possible single-table merge).
        Raises on any fatal error so the caller can log it.
        """
        from napari_flopa.core.io.loader import read_ptu_file
        from napari_flopa.core.processing.reconstruction import (
            reconstruct_ptu_to_dataset,
        )

        stem = ptu_path.stem

        # Determine which variables are needed
        need = set()
        if p.get("image_configs"):
            need |= {"photon_count", "mean_arrival_time"}
        if p.get("phasor_configs"):
            need |= {"phasor", "photon_count"}
        if p.get("decay_configs"):
            need.add("tcspc_histogram")
        outputs = list(need) if need else None

        ptu_data = read_ptu_file(str(ptu_path))
        ds = reconstruct_ptu_to_dataset(
            ptu_data, p["scan_config"], outputs=outputs
        )

        # Apply calibration factor
        cal = complex(p.get("cal_real", 1.0), p.get("cal_imag", 0.0))
        if cal != (1 + 0j) and "phasor_g" in ds and "phasor_s" in ds:
            import xarray as xr

            raw = ds["phasor_g"].values + 1j * ds["phasor_s"].values
            cal_result = raw * cal
            ds = ds.assign(
                {
                    "phasor_g": xr.DataArray(
                        cal_result.real,
                        dims=ds["phasor_g"].dims,
                        coords=ds["phasor_g"].coords,
                    ),
                    "phasor_s": xr.DataArray(
                        cal_result.imag,
                        dims=ds["phasor_s"].dims,
                        coords=ds["phasor_s"].coords,
                    ),
                }
            )

        # Match label files
        lbl_paths: list[Path | None] = [None]
        if lbl_dir and lbl_dir.is_dir():
            matched = [
                f
                for ext in ("*.tif", "*.tiff")
                for f in lbl_dir.glob(ext)
                if f.stem.startswith(stem)
            ]
            if matched:
                lbl_paths = matched

        phasor_file_rows: list[dict] = []

        for cfg in p.get("image_configs", []):
            self._export_images(ds, cfg, stem, out_dir)

        for cfg in p.get("phasor_configs", []):
            rows = self._extract_phasor(ds, cfg, stem, lbl_paths)
            phasor_file_rows.extend(rows)

        if p.get("per_file_table") and phasor_file_rows:
            _write_csv(out_dir / f"{stem}_phasor.csv", phasor_file_rows)

        for cfg in p.get("decay_configs", []):
            rows, headers = self._extract_decay(ds, cfg, stem)
            if rows:
                _write_csv_ordered(
                    out_dir / f"{stem}_decay.csv", headers, rows
                )

        return phasor_file_rows  # returned for single-table merge

    # ── image export ─────────────────────────────────────────────────────

    def _export_images(self, ds, cfg: dict, stem: str, out_dir: Path):
        """Write TIFF images for one aggregation config pass."""
        from matplotlib import cm as _cm

        from napari_flopa.core.processing.image_utils import (
            aggregate_dataset,
            smooth_count,
            smooth_weighted,
        )

        try:
            from skimage.io import imsave
        except ImportError:
            import tifffile

            def imsave(p, a, **kw):
                tifffile.imsave(str(p), a)

        res_ns = float(
            ds.attrs.get("instrument_params", {}).get(
                "tcspc_resolution_ns", 1.0
            )
        )

        # Build iteration combos
        if cfg.get("single_frame"):
            _sh = {"frame": "F", "sequence": "S", "channel": "D"}
            sel = {}
            for dim, key in (
                ("frame", "sf_frame"),
                ("sequence", "sf_seq"),
                ("channel", "sf_det"),
            ):
                if dim in ds.sizes:
                    sel[dim] = min(cfg.get(key, 0), ds.sizes[dim] - 1)
            suffix = "_".join(f"{_sh[k]}{v}" for k, v in sel.items())
            free_combos = [(sel, suffix)]
            sum_dims = []
        else:
            sum_dims, free_combos = _agg_combos(ds, cfg)

        lt_lut = None
        if cfg.get("export_flim") and "mean_arrival_time" in ds:
            cmap_name = cfg.get("cmap", "rainbow")
            lt_lut = (
                _cm.get_cmap(cmap_name)(np.linspace(0, 1, 256))[:, :3] * 255
            ).astype(np.uint8)

        for combo, suffix in free_combos:
            sliced = ds.isel(**combo) if combo else ds
            agg = aggregate_dataset(sliced, sum_dims) if sum_dims else sliced

            ci = (
                agg["photon_count"].values.squeeze()
                if "photon_count" in agg
                else None
            )
            cl = (
                agg["mean_arrival_time"].values.squeeze()
                if "mean_arrival_time" in agg
                else None
            )

            if cfg.get("smooth"):
                k = cfg.get("smooth_kernel", 3)
                if cl is not None and ci is not None:
                    cl, _ = smooth_weighted(cl, ci.astype(np.uint32), size=k)
                if ci is not None:
                    ci = smooth_count(ci, size=k)

            sm_tag = (
                f'_sm{cfg.get("smooth_kernel", 3)}'
                if cfg.get("smooth")
                else ""
            )
            pfx = f"{stem}_{suffix}{sm_tag}" if suffix else f"{stem}{sm_tag}"

            if cfg.get("export_int") and ci is not None:
                mn, mx = float(ci.min()), float(ci.max())
                scaled = ((ci - mn) / max(mx - mn, 1e-9) * 65535).astype(
                    np.uint16
                )
                imsave(
                    out_dir / f"{pfx}_intensity.tif",
                    scaled,
                    check_contrast=False,
                )

            if cfg.get("export_lt") and cl is not None:
                imsave(
                    out_dir / f"{pfx}_lifetime.tif",
                    (cl * res_ns).astype(np.float32),
                    check_contrast=False,
                )

            if (
                cfg.get("export_flim")
                and ci is not None
                and cl is not None
                and lt_lut is not None
            ):
                cl_ns = cl.astype(np.float32) * res_ns

                # Lifetime LUT range — explicit ns values, or auto = 2nd/98th percentile
                # (matches HistogramSlider.update_data default in FlimView)
                lt_ns_lo = cfg.get("lt_ns_lo")
                lt_ns_hi = cfg.get("lt_ns_hi")
                if lt_ns_lo is None or lt_ns_hi is None:
                    cl_valid = cl_ns[np.isfinite(cl_ns)]
                    auto_lo, auto_hi = (
                        np.percentile(cl_valid, [2, 98])
                        if cl_valid.size > 0
                        else (0.0, 1.0)
                    )
                lt_lo = (
                    float(lt_ns_lo) if lt_ns_lo is not None else float(auto_lo)
                )
                lt_hi = (
                    float(lt_ns_hi) if lt_ns_hi is not None else float(auto_hi)
                )
                lt_range = lt_hi - lt_lo if lt_hi > lt_lo else 1.0
                lt_idx = np.clip(
                    (cl_ns - lt_lo) / lt_range * 255, 0, 255
                ).astype(np.uint8)

                # Intensity range — 2nd/98th percentile (matches FlimView default)
                ci_f = ci.astype(np.float32)
                ci_valid = ci_f[np.isfinite(ci_f)]
                int_lo, int_hi = (
                    np.percentile(ci_valid, [2, 98])
                    if ci_valid.size > 0
                    else (float(ci_f.min()), float(ci_f.max()))
                )
                int_range = int_hi - int_lo if int_hi > int_lo else 1.0
                int_norm = np.clip((ci_f - int_lo) / int_range, 0.0, 1.0)

                rgb = (lt_lut[lt_idx].astype(np.float32) / 255.0) * int_norm[
                    ..., np.newaxis
                ]
                imsave(
                    out_dir / f"{pfx}_flim.png",
                    (rgb * 255).clip(0, 255).astype(np.uint8),
                    check_contrast=False,
                )

    # ── phasor extraction ────────────────────────────────────────────────

    def _extract_phasor(
        self, ds, cfg: dict, stem: str, lbl_paths: list
    ) -> list[dict]:
        """Extract phasor summary rows for one aggregation config."""
        if "phasor_g" not in ds or "phasor_s" not in ds:
            return []
        from napari_flopa.core.processing.image_utils import aggregate_dataset

        per_pixel = cfg.get("per_pixel", False)
        rows: list[dict] = []
        sum_dims, free_combos = _agg_combos(ds, cfg)

        smooth = cfg.get("smooth", False)
        smooth_k = cfg.get("smooth_kernel", 3)

        for combo, agg_label in free_combos:
            sliced = ds.isel(**combo) if combo else ds
            agg = aggregate_dataset(sliced, sum_dims) if sum_dims else sliced

            g2d = agg["phasor_g"].values.squeeze()
            s2d = agg["phasor_s"].values.squeeze()
            pc2d = (
                agg["photon_count"].values.squeeze()
                if "photon_count" in agg
                else None
            )

            if smooth and g2d.ndim == 2:
                from scipy.ndimage import uniform_filter

                g2d = uniform_filter(g2d.astype(np.float64), size=smooth_k)
                s2d = uniform_filter(s2d.astype(np.float64), size=smooth_k)

            sm_tag = f"_sm{smooth_k}" if smooth else ""
            agg_label_full = f"{agg_label}{sm_tag}"

            for lbl_path in lbl_paths:
                lbl_arr = None
                lbl_name = "no_mask"
                if lbl_path is not None:
                    try:
                        from skimage.io import imread

                        lbl_arr = imread(str(lbl_path)).astype(np.int32)
                        lbl_name = lbl_path.stem
                        if lbl_arr.shape != g2d.shape:
                            lbl_arr = None
                    except Exception:
                        pass

                label_ids = (
                    [0]
                    if lbl_arr is None
                    else sorted(np.unique(lbl_arr[lbl_arr > 0]))
                )

                for lid in label_ids:
                    mask = (
                        np.ones(g2d.shape, bool)
                        if lbl_arr is None
                        else lbl_arr == lid
                    )
                    # Match phasor panel: exclude non-finite and zero-photon pixels
                    valid = mask & np.isfinite(g2d) & np.isfinite(s2d)
                    if pc2d is not None:
                        valid &= pc2d > 0
                    g_m = g2d[valid].astype(np.float64)
                    s_m = s2d[valid].astype(np.float64)
                    pc_m = (
                        pc2d[valid].astype(np.float64)
                        if pc2d is not None
                        else np.ones(valid.sum())
                    )
                    if g_m.size == 0:
                        continue

                    if per_pixel:
                        for gi, si, pci in zip(g_m, s_m, pc_m, strict=True):
                            rows.append(
                                dict(
                                    ptu_file=stem,
                                    labels_file=lbl_name,
                                    agg=agg_label_full,
                                    label_id=int(lid),
                                    g=float(gi),
                                    s=float(si),
                                    photon_count=float(pci),
                                )
                            )
                    else:
                        w = pc_m
                        ws = w.sum() or 1.0
                        rows.append(
                            dict(
                                ptu_file=stem,
                                labels_file=lbl_name,
                                agg=agg_label_full,
                                label_id=int(lid),
                                g=float((g_m * w).sum() / ws),
                                s=float((s_m * w).sum() / ws),
                                photon_count=float(ws),
                                area_pixels=int(valid.sum()),
                            )
                        )
        return rows

    # ── decay extraction ─────────────────────────────────────────────────

    def _extract_decay(
        self, ds, cfg: dict, stem: str
    ) -> tuple[list[dict], list[str]]:
        """
        Return (rows, headers) for one aggregation config.

        Rows are dicts: ptu_file, agg, time_ns, <curve_label...>
        Headers list preserves column order.
        """
        if "tcspc_histogram" not in ds:
            return [], []
        da = ds["tcspc_histogram"]
        for dim, key in (
            ("frame", "agg_frames"),
            ("sequence", "agg_seqs"),
            ("channel", "agg_dets"),
        ):
            if cfg.get(key) and dim in da.dims:
                da = da.sum(dim)

        time_dim = "tcspc_channel"
        free_dims = [d for d in da.dims if d != time_dim]
        ip = ds.attrs.get("instrument_params", {})
        res_ns = float(ip.get("tcspc_resolution_ns", 1.0))
        n_bins = da.sizes[time_dim]
        time_ns = np.arange(n_bins) * res_ns
        _short = {"frame": "F", "sequence": "S", "channel": "D"}
        shift = cfg.get("shift", 0)
        norm = cfg.get("norm", False)

        # Build aggregation label for this pass
        sum_dims = [
            d
            for d in ("frame", "sequence", "channel")
            if cfg.get(
                {
                    "frame": "agg_frames",
                    "sequence": "agg_seqs",
                    "channel": "agg_dets",
                }[d]
            )
        ]
        agg_lbl = (
            "Sum" + "".join(d[0].upper() for d in sum_dims)
            if sum_dims
            else "raw"
        )

        curves: list[tuple[str, np.ndarray]] = []
        if not free_dims:
            curves.append(("Sum", da.values.flatten().astype(np.float64)))
        else:
            sizes = [da.sizes[d] for d in free_dims]
            for combo in itertools.product(*[range(s) for s in sizes]):
                sel = {
                    d: int(v) for d, v in zip(free_dims, combo, strict=True)
                }
                c = da.isel(**sel).values.flatten().astype(np.float64)
                lbl = " ".join(
                    f"{_short.get(d,'?')}{v}" for d, v in sel.items()
                )
                curves.append((lbl, c))

        processed = []
        for lbl, c in curves:
            c = np.roll(c, shift)
            if norm and c.max() > 0:
                c = c / c.max()
            processed.append((lbl, c))

        curve_cols = [lbl for lbl, _ in processed]
        headers = ["ptu_file", "agg", "time_ns"] + curve_cols
        rows = []
        for i in range(n_bins):
            row = {
                "ptu_file": stem,
                "agg": agg_lbl,
                "time_ns": f"{time_ns[i]:.6f}",
            }
            for lbl, c in processed:
                row[lbl] = f"{c[i]:.4f}"
            rows.append(row)
        return rows, headers


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _agg_combos(ds, cfg: dict) -> tuple[list[str], list[tuple[dict, str]]]:
    """
    Return (sum_dims, free_combos) where free_combos is a list of
    (sel_dict, label_suffix) for every combination of non-aggregated dims.
    """
    sum_dims: list[str] = []
    free: dict[str, list[int]] = {}
    for dim, key in (
        ("frame", "agg_frames"),
        ("sequence", "agg_seqs"),
        ("channel", "agg_dets"),
    ):
        if dim not in ds.sizes:
            continue
        if cfg.get(key):
            sum_dims.append(dim)
        else:
            free[dim] = list(range(ds.sizes[dim]))

    _short = {"frame": "F", "sequence": "S", "channel": "D"}
    if not free:
        suffix = (
            ("Sum" + "".join(d[0].upper() for d in sum_dims))
            if sum_dims
            else ""
        )
        return sum_dims, [({}, suffix)]

    combos = []
    for vals in itertools.product(*free.values()):
        sel = dict(zip(free.keys(), vals, strict=True))
        suffix = "_".join(f"{_short.get(k,'?')}{v}" for k, v in sel.items())
        if sum_dims:
            suffix += "_Sum" + "".join(d[0].upper() for d in sum_dims)
        combos.append((sel, suffix))
    return sum_dims, combos


def _write_csv(path: Path, rows: list[dict]):
    """Write a list of uniform dicts to CSV (fieldnames from first row)."""
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_csv_ordered(path: Path, headers: list[str], rows: list[dict]):
    """Write CSV with explicit column order."""
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# BatchPanel
# ─────────────────────────────────────────────────────────────────────────────


class BatchPanel(QWidget):
    """
    Tab 3 — Batch analysis.

    All .ptu files in a chosen directory are reconstructed with a shared
    ScanConfig + calibration factor.  Enabled export sections run for every
    file; each section supports multiple aggregation passes (one CSV/TIFF set
    per pass per file).

    Scan config + calibration are saved/loaded as JSON.
    Processing runs in a QThread; UI stays responsive.
    """

    def __init__(self, state: FlopaState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer
        self._thread: QThread | None = None
        self._worker: _BatchWorker | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # Scrollable top area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        ilay = QVBoxLayout(inner)
        ilay.setContentsMargins(2, 2, 2, 2)
        ilay.setSpacing(6)
        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

        # ── 1. Directories ─────────────────────────────────────────────
        dir_box = QGroupBox("Directories")
        apply_style(dir_box, S.GROUP_PRIMARY)
        dir_box.setStyleSheet(S.GROUP_PRIMARY)
        dg = QGridLayout(dir_box)
        dg.setSpacing(4)
        dg.addWidget(QLabel("PTU folder:"), 0, 0)
        self._ptu_edit = QLineEdit()
        self._ptu_edit.setPlaceholderText("Folder containing .ptu files…")
        dg.addWidget(self._ptu_edit, 0, 1)
        pb = QPushButton("…")
        pb.setFixedWidth(28)
        pb.clicked.connect(self._browse_ptu)
        dg.addWidget(pb, 0, 2)
        dg.addWidget(QLabel("Labels folder:"), 1, 0)
        self._lbl_edit = QLineEdit()
        self._lbl_edit.setPlaceholderText(
            "Optional — TIFF integer label files matched by PTU name prefix"
        )
        dg.addWidget(self._lbl_edit, 1, 1)
        lb = QPushButton("…")
        lb.setFixedWidth(28)
        lb.clicked.connect(self._browse_lbl)
        dg.addWidget(lb, 1, 2)
        self._recursive_chk = QCheckBox("Include sub-folders")
        dg.addWidget(self._recursive_chk, 2, 1)
        ilay.addWidget(dir_box)

        # ── 2. Scan config + calibration ───────────────────────────────
        cfg_box = QGroupBox("Scan Config && Calibration")
        apply_style(cfg_box, S.GROUP_PRIMARY)
        cfg_box.setStyleSheet(S.GROUP_PRIMARY)
        cfg_hlay = QHBoxLayout(cfg_box)
        cfg_hlay.setSpacing(8)
        cfg_hlay.setContentsMargins(6, 4, 6, 6)

        # ── left: fields ────────────────────────────────────────────
        fields_w = QWidget()
        cg = QGridLayout(fields_w)
        cg.setSpacing(4)
        cg.setContentsMargins(0, 0, 0, 0)

        def _le_int(default: int, w: int = 60) -> QLineEdit:
            e = QLineEdit(str(default))
            e.setMaximumWidth(w)
            e.setValidator(QIntValidator(1, 999999))
            return e

        self._c_frames = _le_int(1)
        self._c_lines = _le_int(256)
        self._c_pixels = _le_int(256)
        self._c_seqs = _le_int(1, 50)
        self._c_maxdet = _le_int(4, 50)

        # accum/seq — comma-separated; single value → all seqs, or one-per-seq
        self._c_accu = QLineEdit("256")
        self._c_accu.setMaximumWidth(110)
        self._c_accu.setValidator(
            QRegularExpressionValidator(
                QRegularExpression(r"\d+(\s*,\s*\d+)*")
            )
        )
        self._c_accu.setToolTip(
            "Number of line accumulations per sequence.\n"
            "• Single integer → same value for every sequence, e.g. 256\n"
            "• Comma-separated list → one value per sequence (count must\n"
            "  match N seqs), e.g. for 3 sequences: 1,3,256\n"
            "Error is raised at run time if count mismatches N seqs."
        )

        for col, (lbl, w) in enumerate(
            [
                ("Frames:", self._c_frames),
                ("Lines:", self._c_lines),
                ("Pixels:", self._c_pixels),
            ]
        ):
            cg.addWidget(QLabel(lbl), 0, col * 2)
            cg.addWidget(w, 0, col * 2 + 1)

        for col, (lbl, w) in enumerate(
            [
                ("N seqs:", self._c_seqs),
                ("Accum/seq:", self._c_accu),
                ("Max detector:", self._c_maxdet),
            ]
        ):
            cg.addWidget(QLabel(lbl), 1, col * 2)
            cg.addWidget(w, 1, col * 2 + 1)

        # Bidirectional row
        self._c_bidir = QCheckBox("Bidirectional")
        self._c_bidir.setStyleSheet("font-weight: normal;")
        self._c_bidir.setToolTip("Enable bidirectional scan correction")
        self._c_bidir_shift = QLineEdit("0.0")
        self._c_bidir_shift.setMaximumWidth(65)
        self._c_bidir_shift.setValidator(QDoubleValidator(-0.2, 0.2, 6))
        self._c_bidir_shift.setToolTip(
            "Phase shift for bidirectional correction (pixels, −0.2 … 0.2)"
        )
        self._c_bidir_shift.setEnabled(False)
        self._c_bidir.toggled.connect(self._c_bidir_shift.setEnabled)
        bidir_lbl = QLabel("Phase shift:")
        bidir_lbl.setStyleSheet("font-weight: normal;")
        cg.addWidget(self._c_bidir, 2, 0, 1, 2)
        cg.addWidget(bidir_lbl, 2, 2)
        cg.addWidget(self._c_bidir_shift, 2, 3)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet(S.SEPARATOR)
        cg.addWidget(sep1, 3, 0, 1, 6)

        cal_lbl = QLabel("Calibration")
        cal_lbl.setStyleSheet(S.MUTED)
        cg.addWidget(cal_lbl, 4, 0, 1, 6)

        self._cal_frep = QLineEdit("40.0")
        self._cal_frep.setMaximumWidth(70)
        self._cal_frep.setValidator(QDoubleValidator(0.001, 9999.0, 6))
        self._cal_frep.setToolTip("Laser/excitation repetition rate in MHz")

        self._cal_factor = QLineEdit("1+0j")
        self._cal_factor.setMaximumWidth(110)
        self._cal_factor.setValidator(
            QRegularExpressionValidator(
                QRegularExpression(
                    r"[-+]?\d*\.?\d*([eE][-+]?\d+)?(\s*[-+]\s*\d*\.?\d*([eE][-+]?\d+)?j)?"
                )
            )
        )
        self._cal_factor.setToolTip(
            "Complex calibration factor in Python notation.\n"
            "Example: 1.0+0.2j  or  0.95-0.05j"
        )

        cg.addWidget(QLabel("f_rep (MHz):"), 5, 0)
        cg.addWidget(self._cal_frep, 5, 1)
        cg.addWidget(QLabel("Factor:"), 5, 2)
        cg.addWidget(self._cal_factor, 5, 3)

        cfg_hlay.addWidget(fields_w, stretch=1)

        # ── right: action buttons ────────────────────────────────────
        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)
        self._load_scan_btn = QPushButton("↓ Scan config")
        self._load_scan_btn.setToolTip(
            "Copy scan config from the currently reconstructed file"
        )
        self._load_cal_btn = QPushButton("↓ Calibration")
        self._load_cal_btn.setToolTip(
            "Copy calibration (f_rep) from the currently reconstructed file"
        )
        for b in (self._load_scan_btn, self._load_cal_btn):
            b.setEnabled(False)
            b.setStyleSheet(S.BTN_SMALL)
            btn_col.addWidget(b)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(S.SEPARATOR)
        btn_col.addWidget(sep2)

        self._load_cfg_btn = QPushButton("Load JSON…")
        self._save_cfg_btn = QPushButton("Save JSON…")
        for b in (self._load_cfg_btn, self._save_cfg_btn):
            b.setStyleSheet(S.BTN_SMALL)
            btn_col.addWidget(b)

        btn_col.addStretch()
        cfg_hlay.addLayout(btn_col)
        ilay.addWidget(cfg_box)

        # ── 3. Export sections ─────────────────────────────────────────
        self._images_sec = _ImagesSection()
        self._phasor_sec = _PhasorSection()
        self._decay_sec = _DecaySection()
        for sec in (self._images_sec, self._phasor_sec, self._decay_sec):
            ilay.addWidget(sec)

        # ── 4. Output options ──────────────────────────────────────────
        out_box = QGroupBox("Output options")
        apply_style(out_box, S.GROUP_PRIMARY)
        out_box.setStyleSheet(S.GROUP_PRIMARY)
        ol = QVBoxLayout(out_box)
        ol.setSpacing(3)
        tr = QHBoxLayout()
        self._per_file_chk = QCheckBox("Per-PTU tables")
        self._single_tbl_chk = QCheckBox("Single merged phasor table")
        self._per_file_chk.setChecked(True)
        tr.addWidget(self._per_file_chk)
        tr.addWidget(self._single_tbl_chk)
        tr.addStretch()
        ol.addLayout(tr)
        self._out_dir_label = QLabel(
            "Output directory: batch_<timestamp>/ inside PTU folder"
        )
        self._out_dir_label.setStyleSheet(S.HINT)
        ol.addWidget(self._out_dir_label)
        ilay.addWidget(out_box)

        ilay.addStretch()

        # ── 5. Run controls ────────────────────────────────────────────
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Batch")
        self._run_btn.setStyleSheet(S.BTN_RUN)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(S.BTN_STOP)
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._stop_btn)
        run_row.addStretch()
        root.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(130)
        self._log.setStyleSheet(S.LOG)
        root.addWidget(self._log)

        # Wiring
        self._run_btn.clicked.connect(self._on_run)
        self._stop_btn.clicked.connect(self._on_stop)
        self._load_cfg_btn.clicked.connect(self._on_load_json)
        self._save_cfg_btn.clicked.connect(self._on_save_json)
        self._load_scan_btn.clicked.connect(self._on_load_scan)
        self._load_cal_btn.clicked.connect(self._on_load_calibration)
        self.state.dataset_changed.connect(self._on_dataset_changed)

    # ── directory pickers ────────────────────────────────────────────────

    def _browse_ptu(self):
        d = QFileDialog.getExistingDirectory(self, "Select PTU folder")
        if d:
            self._ptu_edit.setText(d)

    def _browse_lbl(self):
        d = QFileDialog.getExistingDirectory(self, "Select Labels folder")
        if d:
            self._lbl_edit.setText(d)

    # ── Config JSON ──────────────────────────────────────────────────────

    def _config_dict(self) -> dict:
        """Collect scan config + calibration into a serialisable dict."""
        return {
            "scan": dict(
                frames=int(self._c_frames.text() or 1),
                lines=int(self._c_lines.text() or 256),
                pixels=int(self._c_pixels.text() or 256),
                sequences=int(self._c_seqs.text() or 1),
                accum_per_seq=self._c_accu.text().strip() or "1",
                max_detector=int(self._c_maxdet.text() or 4),
                bidirectional=self._c_bidir.isChecked(),
                bidirectional_phase_shift=float(
                    self._c_bidir_shift.text() or 0.0
                ),
            ),
            "calibration": dict(
                f_rep_mhz=float(self._cal_frep.text() or 40.0),
                factor=self._cal_factor.text().strip() or "1+0j",
            ),
        }

    def _apply_dict(self, cfg: dict):
        """Push a config dict back into the UI."""
        s = cfg.get("scan", {})
        self._c_frames.setText(str(s.get("frames", 1)))
        self._c_lines.setText(str(s.get("lines", 256)))
        self._c_pixels.setText(str(s.get("pixels", 256)))
        self._c_seqs.setText(str(s.get("sequences", 1)))
        self._c_accu.setText(str(s.get("accum_per_seq", "1")))
        self._c_maxdet.setText(str(s.get("max_detector", 4)))
        self._c_bidir.setChecked(bool(s.get("bidirectional", False)))
        self._c_bidir_shift.setText(
            str(s.get("bidirectional_phase_shift", 0.0))
        )
        c = cfg.get("calibration", {})
        self._cal_frep.setText(str(c.get("f_rep_mhz", 40.0)))
        self._cal_factor.setText(str(c.get("factor", "1+0j")))

    def _on_load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self._apply_dict(load_config(path))
            self._log_line(f"Loaded: {Path(path).name}")
        except Exception as e:
            self._log_line(f"Load error: {e}", error=True)

    def _on_save_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", "batch_config.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            save_config(path, self._config_dict())
            self._log_line(f"Saved: {Path(path).name}")
        except Exception as e:
            self._log_line(f"Save error: {e}", error=True)

    # ── run ──────────────────────────────────────────────────────────────

    def _build_scan_config(self):
        """Build a ScanConfig from the UI fields."""
        from tttrkit.ptuio.reconstructor import ScanConfig

        def _int(edit: QLineEdit, name: str, default: int = 1) -> int:
            txt = edit.text().strip()
            try:
                return int(txt) if txt else default
            except ValueError:
                raise ValueError(
                    f"{name}: expected integer, got {txt!r}"
                ) from None

        n_seqs = _int(self._c_seqs, "N seqs", 1)

        # Parse accum/seq — single value or comma-separated list
        accum_parts = [
            p.strip() for p in self._c_accu.text().split(",") if p.strip()
        ]
        if not accum_parts:
            accum_parts = ["1"]
        try:
            accum_vals = [int(p) for p in accum_parts]
        except ValueError as exc:
            raise ValueError(f"Accum/seq: {exc}") from exc

        if len(accum_vals) == 1:
            line_accumulations = tuple([accum_vals[0]] * n_seqs)
        elif len(accum_vals) == n_seqs:
            line_accumulations = tuple(accum_vals)
        else:
            raise ValueError(
                f"Accum/seq: got {len(accum_vals)} values but N seqs = {n_seqs}. "
                f"Provide 1 value (applied to all) or exactly {n_seqs} values."
            )

        try:
            bidir_shift = float(self._c_bidir_shift.text() or 0.0)
        except ValueError:
            bidir_shift = 0.0

        return ScanConfig(
            frames=_int(self._c_frames, "Frames"),
            pixels=_int(self._c_pixels, "Pixels"),
            lines=_int(self._c_lines, "Lines"),
            max_detector=_int(self._c_maxdet, "Max detector", 4),
            line_accumulations=line_accumulations,
            bidirectional=self._c_bidir.isChecked(),
            bidirectional_phase_shift=bidir_shift,
            frame_start_marker_channel=4,
            line_start_marker_channel=1,
            line_stop_marker_channel=2,
        )

    def _cal_complex(self) -> complex:
        """Parse the calibration factor field."""
        txt = self._cal_factor.text().strip().replace(" ", "")
        if not txt:
            return 1 + 0j
        try:
            return complex(txt)
        except ValueError:
            raise ValueError(
                f"Calibration factor: invalid complex number {txt!r}"
            ) from None

    def _on_run(self):
        ptu_dir = self._ptu_edit.text().strip()
        if not ptu_dir or not Path(ptu_dir).is_dir():
            self._log_line("Select a valid PTU folder first.", error=True)
            return

        img_cfgs = self._images_sec.get_configs()
        phasor_cfgs = self._phasor_sec.get_configs()
        decay_cfgs = self._decay_sec.get_configs()
        if not any([img_cfgs, phasor_cfgs, decay_cfgs]):
            self._log_line("Enable at least one export section.", error=True)
            return

        try:
            scan_cfg = self._build_scan_config()
            cal = self._cal_complex()
        except Exception as e:
            self._log_line(f"Config error: {e}", error=True)
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(ptu_dir) / f"batch_{ts}"
        self._out_dir_label.setText(f"Output: {out_dir}")

        params = dict(
            ptu_dir=ptu_dir,
            lbl_dir=self._lbl_edit.text().strip() or None,
            recursive=self._recursive_chk.isChecked(),
            scan_config=scan_cfg,
            cal_real=cal.real,
            cal_imag=cal.imag,
            image_configs=img_cfgs,
            phasor_configs=phasor_cfgs,
            decay_configs=decay_cfgs,
            per_file_table=self._per_file_chk.isChecked(),
            single_table=self._single_tbl_chk.isChecked(),
            out_dir=out_dir,
        )

        self._thread = QThread()
        self._worker = _BatchWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._log.appendHtml)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._log.clear()
        self._thread.start()

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self._stop_btn.setEnabled(False)

    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        self._progress.setRange(0, total)
        self._progress.setValue(current)

    @Slot(str)
    def _on_finished(self, _msg: str):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)

    # ── load-from-current helpers ────────────────────────────────────────

    def _on_dataset_changed(self):
        has_data = self.state.has_data()
        self._load_scan_btn.setEnabled(has_data)
        self._load_cal_btn.setEnabled(has_data)

    def _on_load_scan(self):
        ds = self.state.dataset
        if ds is None:
            return
        sc = ds.attrs.get("scan_config", {})
        if not sc:
            self._log_line(
                "No scan config stored in current dataset.", error=True
            )
            return
        accum = sc.get("line_accumulations", (1,))
        self._c_frames.setText(str(sc.get("frames", 1)))
        self._c_lines.setText(str(sc.get("lines", 256)))
        self._c_pixels.setText(str(sc.get("pixels", 256)))
        self._c_seqs.setText(str(len(accum)))
        self._c_accu.setText(",".join(str(a) for a in accum))
        self._c_maxdet.setText(str(sc.get("max_detector", 4)))
        self._c_bidir.setChecked(bool(sc.get("bidirectional", False)))
        self._c_bidir_shift.setText(
            str(sc.get("bidirectional_phase_shift", 0.0))
        )
        self._log_line("Scan config loaded from current dataset.")

    def _on_load_calibration(self):
        factor = self.state.calib_factor
        self._cal_factor.setText(
            f"{factor.real:.6g}+{factor.imag:.6g}j"
            if factor.imag >= 0
            else f"{factor.real:.6g}{factor.imag:.6g}j"
        )
        self._log_line(f"Calibration factor loaded: {factor}")

    def _log_line(self, msg: str, *, error: bool = False):
        self._log.appendHtml((_ERR if error else _INFO).format(msg))
