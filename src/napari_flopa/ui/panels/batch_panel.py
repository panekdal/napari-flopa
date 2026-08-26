"""
Batch analysis panel.

Reconstructs a directory of .ptu files — or a hand-picked subset of it — with
a shared ScanConfig + calibration factor and writes any combination of:
  • FLIM RGB / Intensity / Lifetime  TIFF images
  • Phasor CSV tables  (per-object or per-pixel)
  • Decay CSV tables

Each export type has independent aggregation passes: add multiple rows to
run e.g. "no aggregation" AND "sum-frames + sum-detectors" in one batch.

Labels (optional): integer TIFFs in a separate folder matched by PTU name
prefix (0 = background).  Multiple label files per PTU are merged into one
per-file table; 'labels_file' is a column in every row.

Output: written to  <ptu_dir>/batch_<timestamp>/
Processing: sequential in a QThread — UI stays responsive; Stop aborts
between files.

Scan config + calibration can be saved/loaded.
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

from napari_flopa.core.io.config import (
    DEFAULT_LASER_DUTY,
    ScanSettings,
    load_config,
    save_config,
)
from napari_flopa.core.processing.image_utils import (
    LIFETIME_COLORMAPS,
    auto_range,
    colormap_to_lut,
    flim_export_info,
    flim_rgb,
)
from napari_flopa.core.processing.reconstruction import (
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
)
from napari_flopa.ui.state import FlopaState
from napari_flopa.ui.style import S, apply_style

# ── log colour helpers ────────────────────────────────────────────────────
_ERR = '<span style="color:#ff6060">{}</span>'
_OK = '<span style="color:#60cc60">{}</span>'
_INFO = '<span style="color:#aaaaaa">{}</span>'


class _AggRow(QWidget):
    """
    One aggregation pass: F / S / D checkboxes plus kind-specific extras.

    kind='images'  → no extras
    kind='phasor'  → no extras
    kind='decay'   → + Norm checkbox


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

        if kind == "decay":
            lay.addWidget(_vsep())
            self.norm_chk = QCheckBox("Norm")
            self.norm_chk.setToolTip(
                "Normalise each curve of this set to its own peak"
            )
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
        if self.norm_chk is not None:
            d["norm"] = self.norm_chk.isChecked()
        return d


def _vsep() -> QFrame:
    # vertical separator (line)
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setStyleSheet(S.SEPARATOR)
    return f


def _needed_outputs(p: dict) -> list[str] | None:
    """Reconstruction outputs the enabled exports actually consume.

    The same three levels the File tab offers.
    ``None`` means "compute all outputs".

    """
    if p.get("phasor_configs") or p.get("decay_configs"):
        return None

    img = p.get("image_configs") or []
    if any(c.get("export_lt") or c.get("export_flim") for c in img):
        # FLIM RGB composites counts × lifetime, so both are needed.
        return ["photon_count", "mean_arrival_time"]
    if any(c.get("export_int") for c in img):
        return ["photon_count"]
    return None


def _follow_checkbox(checkbox: QCheckBox, *widgets: QWidget):
    """Enable *widgets* only while *checkbox* is ticked.

    Labels are included on purpose: Qt dims a disabled widget's own text, so a
    field greys out but its caption would stay bright and the row would read as
    half-active. Applies the current state immediately.
    """
    for widget in widgets:
        widget.setEnabled(checkbox.isChecked())
        checkbox.toggled.connect(widget.setEnabled)


def _first_set(typed, derived) -> float:
    """The value the user typed, or the one derived from the image."""
    return float(derived if typed is None else typed)


def _transparent(widget: QWidget, name: str) -> QWidget:
    """Let the surrounding group box's tint show through a container widget."""
    widget.setObjectName(name)
    widget.setStyleSheet(f"QWidget#{name} {{ background: transparent; }}")
    return widget


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
        lay.setSpacing(4)

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

        add_btn = QPushButton("+")
        add_btn.setFixedWidth(28)
        add_btn.setToolTip("Add another export pass")
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
        super().__init__("Images", kind="images", parent=parent)
        # Apply the initial widget states.
        self._on_mode_toggled(self._sf_radio.isChecked())
        self._on_flim_toggled(self._flim_chk.isChecked())

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
        lbl2 = QLabel("Lifetime (Lt.) colormap:")
        lbl2.setStyleSheet(S.MUTED)
        row2.addWidget(lbl2)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(LIFETIME_COLORMAPS)
        # self._cmap_combo.setMaximumWidth(90)
        self._cmap_combo.setStyleSheet("font-weight: normal;")
        row2.addWidget(self._cmap_combo)
        row2.addStretch()
        lay.addLayout(row2)

        # Row 3 — the two LUT ranges, side by side under the colormap.
        row3 = QHBoxLayout()
        row3.setSpacing(3)
        self._lt_lo = QLineEdit()
        self._lt_hi = QLineEdit()
        self._int_lo = QLineEdit()
        self._int_hi = QLineEdit()
        for e, which, unit in (
            (self._lt_lo, "Minimum", "min"),
            (self._lt_hi, "Maximum", "max"),
        ):
            e.setValidator(QDoubleValidator(0.0, 9999.0, 4))
            e.setToolTip(
                f"{which} lifetime for the FLIM RGB LUT\n"
                "ns when the header carries the TCSPC resolution, raw channels otherwise.\n"
                f"Empty = auto ({unit} of each exported image)."
            )
        for e, which, unit in (
            (self._int_lo, "Minimum", "min"),
            (self._int_hi, "Maximum", "max"),
        ):
            e.setValidator(QDoubleValidator(0.0, 1e12, 4))
            e.setToolTip(
                f"{which} photon count scaling the FLIM RGB brightness — "
                f"empty = auto ({unit} of each exported image)"
            )
        for text, lo, hi in (
            ("Lt. range:", self._lt_lo, self._lt_hi),
            ("Int. range:", self._int_lo, self._int_hi),
        ):
            lbl = QLabel(text)
            lbl.setStyleSheet(S.MUTED)
            row3.addWidget(lbl)
            for e in (lo, hi):
                # e.setMaximumWidth(55)
                e.setPlaceholderText("auto")
            row3.addWidget(lo)
            row3.addWidget(QLabel("–"))
            row3.addWidget(hi)
            row3.addSpacing(3)
        row3.addStretch()
        lay.addLayout(row3)
        # Colormap and both ranges only shape the FLIM RGB composite.
        self._flim_chk.toggled.connect(self._on_flim_toggled)

        # Smoothing
        row_sm = QHBoxLayout()
        row_sm.setSpacing(3)
        self._sm_int_chk = QCheckBox("Smooth Int.")
        self._sm_int_spin = self._kernel_spin()
        self._sm_lt_chk = QCheckBox("Smooth Lt.")
        self._sm_lt_spin = self._kernel_spin()
        for chk, spin in (
            (
                self._sm_int_chk,
                self._sm_int_spin,
            ),
            (
                self._sm_lt_chk,
                self._sm_lt_spin,
            ),
        ):
            chk.setStyleSheet("font-weight: normal;")
            spin.setEnabled(False)
            chk.toggled.connect(spin.setEnabled)
            row_sm.addWidget(chk)
            row_sm.addWidget(spin)
            row_sm.addSpacing(24)
        row_sm.addStretch()
        lay.addLayout(row_sm)

        # Export mode: aggregation passes OR single-frame pick
        row3 = QHBoxLayout()
        self._mode_grp = QButtonGroup(self)
        self._all_radio = QRadioButton("Configure sets")
        self._sf_radio = QRadioButton("Single slice")
        self._sf_radio.setChecked(True)
        for r in (self._sf_radio, self._all_radio):
            self._mode_grp.addButton(r)
            r.setStyleSheet("font-weight: normal;")
            row3.addWidget(r)
        row3.addStretch()
        lay.addLayout(row3)

        # Single-slice selector; its visibility follows the radio above
        self._sf_widget = _transparent(QWidget(), "sfRow")
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
        lay.addWidget(self._sf_widget)

        self._sf_radio.toggled.connect(self._on_mode_toggled)

    @staticmethod
    def _kernel_spin() -> QSpinBox:
        """Odd-sized kernel spin box, as in the FLIM View / Phasor tabs."""
        sp = QSpinBox()
        sp.setRange(3, 19)
        sp.setSingleStep(2)
        sp.setValue(3)
        sp.setFixedWidth(46)
        sp.setStyleSheet("font-weight: normal;")
        return sp

    def set_dims(self, frames: int, sequences: int, detectors: int):
        """Cap the single-slice pickers at what the scan config declares."""
        for spin, n in (
            (self._sf_frame, frames),
            (self._sf_seq, sequences),
            (self._sf_det, detectors),
        ):
            spin.setMaximum(max(0, int(n) - 1))

    def _on_flim_toggled(self, on: bool):
        """Colormap and ranges are diasbled unless RGB composite is wanted."""
        for w in (
            self._cmap_combo,
            self._lt_lo,
            self._lt_hi,
            self._int_lo,
            self._int_hi,
        ):
            w.setEnabled(on)

    def _on_mode_toggled(self, single: bool):
        self._sf_widget.setVisible(single)
        self._rows_widget.setEnabled(not single)

    def get_configs(self) -> list[dict]:
        if not self.isChecked():
            return []
        base = self._type_config()
        if base.get("single_frame"):
            return [base]
        return [{**base, **row.to_dict()} for row in self._rows]

    def _type_config(self):
        def _opt(edit) -> float | None:
            """Typed value, or None meaning 'derive it from the image'."""
            txt = edit.text().strip()
            return float(txt) if txt else None

        cfg = dict(
            export_flim=self._flim_chk.isChecked(),
            export_int=self._int_chk.isChecked(),
            export_lt=self._lt_chk.isChecked(),
            lt_ns_lo=_opt(self._lt_lo),
            lt_ns_hi=_opt(self._lt_hi),
            int_lo=_opt(self._int_lo),
            int_hi=_opt(self._int_hi),
            cmap=self._cmap_combo.currentText(),
            # Section-level, so single-slice and every set share them.
            smooth_int=self._sm_int_chk.isChecked(),
            smooth_int_k=self._sm_int_spin.value(),
            smooth_lt=self._sm_lt_chk.isChecked(),
            smooth_lt_k=self._sm_lt_spin.value(),
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
        self._smooth_chk = QCheckBox("Smooth phasor")
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(3, 19)
        self._smooth_spin.setSingleStep(2)
        self._smooth_spin.setValue(3)
        self._smooth_spin.setFixedWidth(46)
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
    """Export decay CSV: time + one column per curve (always per-file)."""

    def __init__(self, parent=None):
        super().__init__("Decay table (CSV)", kind="decay", parent=parent)


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

        # An explicit pick wins over globbing the folder.
        explicit = [Path(f) for f in p.get("ptu_files") or []]
        if explicit:
            ptu_files = explicit
        else:
            glob = "**/*.ptu" if p.get("recursive") else "*.ptu"
            ptu_files = sorted(ptu_dir.glob(glob))
        if not ptu_files:
            self.log.emit(_ERR.format("No .ptu files found."))
            self.finished.emit("No files processed.")
            return

        wanted = _needed_outputs(p)
        self.log.emit(
            _INFO.format(
                "Reconstructing: "
                + (", ".join(wanted) if wanted else "all outputs")
            )
        )

        n_ok = n_err = 0
        phasor_merged: list[dict] = []
        # Merged decay grows sideways: one column per file × set, sharing the
        # first file's time axis.
        decay_time = np.empty(0)
        decay_merged: dict[str, np.ndarray] = {}

        for i, path in enumerate(ptu_files):
            if self._stop:
                self.log.emit(_INFO.format("Stopped by user."))
                break
            self.progress.emit(i, len(ptu_files))
            self.log.emit(
                _INFO.format(f"[{i+1}/{len(ptu_files)}]  {path.name}")
            )
            try:
                ph_rows, d_time, d_cols = self._process_one(
                    path, lbl_dir, out_dir, p
                )
                phasor_merged.extend(ph_rows)
                if d_cols:
                    if decay_time.size == 0:
                        decay_time = d_time
                    elif d_time.size != decay_time.size:
                        self.log.emit(
                            _ERR.format(
                                f"  ! {path.name}: {d_time.size} TCSPC bins vs "
                                f"{decay_time.size} in the merged table — "
                                "columns padded/truncated to match"
                            )
                        )
                    n = decay_time.size
                    for lbl, counts in d_cols.items():
                        col = np.full(n, np.nan)
                        col[: min(n, counts.size)] = counts[:n]
                        decay_merged[f"{path.stem}_{lbl}"] = col
                n_ok += 1
                self.log.emit(_OK.format("  ✓"))
            except Exception as exc:
                n_err += 1
                self.log.emit(_ERR.format(f"  ✗  {exc}"))
                traceback.print_exc()

        self.progress.emit(len(ptu_files), len(ptu_files))

        # Single merged phasor table — rows from every file stacked
        if p.get("single_table") and phasor_merged and p.get("phasor_configs"):
            _write_csv(out_dir / "phasor_all.csv", phasor_merged)
            self.log.emit(
                _OK.format(
                    f"Merged phasor table: {out_dir / 'phasor_all.csv'}"
                )
            )

        # Single merged decay table — one col per file × set,
        # named <file>_<label>, all against the same time axis.
        if p.get("single_table") and decay_merged:
            _write_decay_csv(
                out_dir / "decay_all.csv", decay_time, decay_merged
            )
            self.log.emit(
                _OK.format(
                    f"Merged decay table: {out_dir / 'decay_all.csv'} "
                    f"({len(decay_merged)} curves)"
                )
            )

        summary = (
            f"Done: {n_ok} succeeded, {n_err} failed.  Output → {out_dir}"
        )
        self.log.emit((_OK if n_err == 0 else _ERR).format(summary))
        self.finished.emit(summary)

    def _process_one(
        self, ptu_path: Path, lbl_dir, out_dir: Path, p: dict
    ) -> tuple[list[dict], np.ndarray, dict[str, np.ndarray]]:
        """
        Reconstruct one PTU file, apply calibration, run all export passes.

        Returns ``(phasor_rows, decay_time_ns, decay_columns)`` — the pieces
        the caller may still merge across files.
        Raises on any fatal error so the caller can log it.
        """
        from napari_flopa.core.io.loader import read_ptu_file
        from napari_flopa.core.processing.reconstruction import (
            reconstruct_ptu_to_dataset,
        )

        stem = ptu_path.stem

        outputs = _needed_outputs(p)

        ptu_data = read_ptu_file(str(ptu_path))
        ds = reconstruct_ptu_to_dataset(
            ptu_data,
            p["scan_config"],
            outputs=outputs,
            tcspc_channels_override=p.get("tcspc_bins"),
            chunk_size=p.get("chunk_size") or DEFAULT_CHUNK_SIZE,
        )
        constants = dict(ptu_data.get("constants") or {})
        if p.get("tcspc_bins"):
            constants["tcspc_bins"] = p["tcspc_bins"]
        ds.attrs["instrument_params"] = constants
        ds.attrs["scan_config"] = p.get("scan_config_dict") or {}
        ds.attrs["source_filename"] = ptu_path.name

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

        # Match label files — searched recursively with the same
        # 'Include sub-folders' switch that selects the PTU files.
        lbl_paths: list[Path | None] = [None]
        if lbl_dir and lbl_dir.is_dir():
            pattern = "**/*.tif" if p.get("recursive") else "*.tif"
            matched = sorted(
                f
                for ext in (pattern, pattern + "f")
                for f in lbl_dir.glob(ext)
                if f.stem.startswith(stem)
            )
            if matched:
                lbl_paths = matched

        phasor_file_rows: list[dict] = []

        for cfg in p.get("image_configs", []):
            self._export_images(ds, cfg, stem, out_dir)

        for cfg in p.get("phasor_configs", []):
            rows = self._extract_phasor(ds, cfg, stem, lbl_paths, lbl_dir)
            phasor_file_rows.extend(rows)

        if p.get("per_file_table") and phasor_file_rows:
            _write_csv(out_dir / f"{stem}_phasor.csv", phasor_file_rows)

        decay_time = np.empty(0)
        decay_cols: dict[str, np.ndarray] = {}
        for cfg in p.get("decay_configs", []):
            t_ns, curves = self._extract_decay(ds, cfg)
            if not curves:
                continue
            if decay_time.size == 0:
                decay_time = t_ns
            for lbl, counts in curves.items():
                if lbl in decay_cols:
                    self.log.emit(
                        _INFO.format(f"  duplicate decay column {lbl} skipped")
                    )
                    continue
                decay_cols[lbl] = counts
        if decay_cols and p.get("per_file_table"):
            _write_decay_csv(
                out_dir / f"{stem}_decay.csv", decay_time, decay_cols, stem
            )

        return phasor_file_rows, decay_time, decay_cols

    # image export
    def _export_images(self, ds, cfg: dict, stem: str, out_dir: Path):
        """Write TIFF images for one aggregation config pass."""
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

        ip = ds.attrs.get("instrument_params", {})
        res_ns = float(ip.get("tcspc_resolution_ns", 1.0))
        lt_unit = ip.get("resolution_unit", "ch")

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
            lt_lut = colormap_to_lut(cfg.get("cmap", "rainbow"))

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

            sm_tag = ""
            if cfg.get("smooth_lt") and cl is not None and ci is not None:
                k_lt = cfg.get("smooth_lt_k", 3)
                cl, _ = smooth_weighted(cl, ci.astype(np.uint32), size=k_lt)
                sm_tag += f"_smLt{k_lt}"
            if cfg.get("smooth_int") and ci is not None:
                k_int = cfg.get("smooth_int_k", 3)
                ci = smooth_count(ci, size=k_int)
                sm_tag += f"_smInt{k_int}"

            pfx = f"{stem}_{suffix}{sm_tag}" if suffix else f"{stem}{sm_tag}"

            if cfg.get("export_int") and ci is not None:
                counts = np.rint(ci).astype(np.uint32)
                imsave(
                    out_dir / f"{pfx}_intensity.tif",
                    counts,
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
                ci_f = ci.astype(np.float32)
                lt_auto = auto_range(cl_ns)
                int_auto = auto_range(ci_f)
                lt_lo = _first_set(cfg.get("lt_ns_lo"), lt_auto[0])
                lt_hi = _first_set(cfg.get("lt_ns_hi"), lt_auto[1])
                int_lo = _first_set(cfg.get("int_lo"), int_auto[0])
                int_hi = _first_set(cfg.get("int_hi"), int_auto[1])

                rgb = flim_rgb(
                    ci_f, cl_ns, lt_lut, (lt_lo, lt_hi), (int_lo, int_hi)
                )
                imsave(
                    out_dir / f"{pfx}_flim.png",
                    (rgb * 255).clip(0, 255).astype(np.uint8),
                    check_contrast=False,
                )

                (out_dir / f"{pfx}_flim.txt").write_text(
                    flim_export_info(
                        cfg.get("cmap", "rainbow"),
                        (lt_lo, lt_hi),
                        (int_lo, int_hi),
                        unit=lt_unit,
                    ),
                    encoding="utf-8",
                )

    # phasor extraction

    def _extract_phasor(
        self, ds, cfg: dict, stem: str, lbl_paths: list, lbl_dir=None
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

            smoothed = False
            if smooth and g2d.ndim == 2 and pc2d is not None:
                from tttrkit.ptuio.utils import smooth_phasor

                k = smooth_k + 1 if smooth_k % 2 == 0 else smooth_k
                phasor_c = smooth_phasor(
                    g2d + 1j * s2d, pc2d.astype(np.uint32), size=k
                )
                g2d, s2d = phasor_c.real, phasor_c.imag
                smooth_k, smoothed = k, True
            elif smooth:
                self.log.emit(
                    _ERR.format(
                        "  ! smoothing skipped — needs photon_count as weights"
                    )
                )

            sm_tag = f"_sm{smooth_k}" if smoothed else ""
            agg_label_full = f"{agg_label}{sm_tag}"

            for lbl_path in lbl_paths:
                lbl_arr = None
                lbl_name = "no_mask"
                if lbl_path is not None:
                    try:
                        from skimage.io import imread

                        lbl_arr = imread(str(lbl_path)).astype(np.int32)
                        # ("mask1.tif" vs "sub/mask1.tif").
                        lbl_name = (
                            lbl_path.relative_to(lbl_dir).as_posix()
                            if lbl_dir is not None
                            else lbl_path.name
                        )
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

    # decay extraction

    def _extract_decay(
        self, ds, cfg: dict
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Curves for one aggregation set, as ``(time_ns, {label: counts})``.

        Every label names all three dimensions the dataset has, with ``∑`` for
        the aggregated ones — ``F0S0D1``, ``F∑S0D0`` — so columns from
        different sets stay distinguishable when merged into one table.
        Normalisation is per set, applied to each curve's own peak.
        """
        if "tcspc_histogram" not in ds:
            return np.empty(0), {}
        da = ds["tcspc_histogram"]

        _short = {"frame": "F", "sequence": "S", "channel": "D"}
        agg_keys = {
            "frame": "agg_frames",
            "sequence": "agg_seqs",
            "channel": "agg_dets",
        }

        present = [d for d in ("frame", "sequence", "channel") if d in da.dims]
        summed = [d for d in present if cfg.get(agg_keys[d])]
        for dim in summed:
            da = da.sum(dim)
        free_dims = [d for d in present if d not in summed]

        ip = ds.attrs.get("instrument_params", {})
        res_ns = float(ip.get("tcspc_resolution_ns", 1.0))
        time_ns = np.arange(da.sizes["tcspc_channel"]) * res_ns
        norm = cfg.get("norm", False)

        def _label(sel: dict) -> str:
            parts = []
            for d in present:
                parts.append(
                    f"{_short[d]}∑" if d in summed else f"{_short[d]}{sel[d]}"
                )
            return "".join(parts)

        curves: dict[str, np.ndarray] = {}
        combos = (
            [{}]
            if not free_dims
            else [
                dict(zip(free_dims, vals, strict=True))
                for vals in itertools.product(
                    *[range(da.sizes[d]) for d in free_dims]
                )
            ]
        )
        for sel in combos:
            c = da.isel(**sel).values.flatten().astype(np.float64)
            if norm and c.max() > 0:
                c = c / c.max()
            curves[_label(sel)] = c
        return time_ns, curves


# Helpers


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


def _write_decay_csv(
    path: Path,
    time_ns: np.ndarray,
    columns: dict[str, np.ndarray],
    ptu_file: str | None = None,
):
    """Write decay curves as columns against a shared time axis.

    ``ptu_file`` adds a leading constant column, used for the per-file tables;
    the merged table leaves it out because the file name is already in every
    column header. Written as utf-8-sig so Excel renders the ``∑`` in labels.
    """
    if not columns or time_ns.size == 0:
        return
    headers = (["ptu_file"] if ptu_file else []) + ["time_ns", *columns]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(time_ns.size):
            row = [ptu_file] if ptu_file else []
            row.append(f"{time_ns[i]:.6f}")
            for counts in columns.values():
                v = counts[i] if i < counts.size else np.nan
                row.append("" if np.isnan(v) else f"{v:.4f}")
            w.writerow(row)


# BatchPanel


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
        self._ptu_dir: Path | None = None
        self._selected_files: list[Path] = []
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
        dg.setSpacing(3)
        dg.addWidget(QLabel("PTU input:"), 0, 0)
        input_row = QHBoxLayout()
        input_row.setSpacing(6)
        self._pick_dir_btn = QPushButton("Select folder…")
        self._pick_dir_btn.setToolTip("Process every .ptu file in a folder")
        self._pick_dir_btn.clicked.connect(self._browse_ptu)
        self._pick_files_btn = QPushButton("Select files…")
        self._pick_files_btn.setToolTip(
            "Process only the .ptu files you pick out of a folder"
        )
        self._pick_files_btn.clicked.connect(self._browse_ptu_files)
        for b in (self._pick_dir_btn, self._pick_files_btn):
            input_row.addWidget(b)
        self._input_lbl = QLabel()
        self._input_lbl.setStyleSheet(S.HINT)
        self._input_lbl.setWordWrap(True)
        input_row.addWidget(self._input_lbl, 0)
        dg.addLayout(input_row, 0, 1)
        dg.addWidget(QLabel("Labels folder:"), 1, 0)
        lbl_row = QHBoxLayout()
        lbl_row.setSpacing(6)
        lb = QPushButton("…")
        lb.setFixedWidth(28)
        lb.clicked.connect(self._browse_lbl)
        lbl_row.addWidget(lb)
        self._lbl_edit = QLineEdit()
        # self._lbl_edit.setMaximumWidth(260)
        self._lbl_edit.setPlaceholderText("Optional — labels…")
        self._lbl_edit.setToolTip(
            "Optional — TIFF integer label files matched by PTU name prefix"
        )
        lbl_row.addWidget(self._lbl_edit)

        # lbl_row.addStretch()
        dg.addLayout(lbl_row, 1, 1)
        self._recursive_chk = QCheckBox("Include sub-folders")
        self._recursive_chk.toggled.connect(self._update_input_label)
        dg.addWidget(self._recursive_chk, 2, 1)
        ilay.addWidget(dir_box)

        # ── 2. Scan config + calibration ───────────────────────────────
        cfg_box = QGroupBox("Scan configuration && Calibration")
        apply_style(cfg_box, S.GROUP_PRIMARY)
        cfg_vlay = QVBoxLayout(cfg_box)
        cfg_vlay.setSpacing(10)
        cfg_vlay.setContentsMargins(6, 4, 6, 6)

        # ── fields ──────────────────────────────────────────────────
        cg = QGridLayout()
        cg.setSpacing(3)
        cg.setContentsMargins(0, 2, 0, 2)

        def _le_int(default: int) -> QLineEdit:
            e = QLineEdit(str(default))
            e.setValidator(QIntValidator(1, 999999))
            return e

        self._c_frames = _le_int(1)
        self._c_lines = _le_int(512)
        self._c_pixels = _le_int(512)
        self._c_seqs = _le_int(1)
        self._c_maxdet = _le_int(1)

        # accum/seq — comma-separated; single value → all seqs, or one-per-seq
        self._c_accu = QLineEdit("1")
        # self._c_accu.setMaximumWidth(110)
        self._c_accu.setValidator(
            QRegularExpressionValidator(
                QRegularExpression(r"\d+(\s*,\s*\d+)*")
            )
        )
        self._c_accu.setToolTip(
            "Number of line accumulations per sequence.\n"
            "• Single integer → same value for every sequence, e.g. 1\n"
            "• Comma-separated list → one value per sequence (count must\n"
            "  match N seqs), e.g. for 3 sequences: 1,3,3\n"
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
                ("Detectors:", self._c_maxdet),
            ]
        ):
            cg.addWidget(QLabel(lbl), 1, col * 2)
            cg.addWidget(w, 1, col * 2 + 1)

        self._c_tcspc = _le_int(1000)
        # self._c_tcspc.setValidator(QIntValidator(1, 1_048_576))
        self._c_tcspc.setToolTip(
            "TCSPC bins (histogram channels). Overrides the value read from "
            "each file's header."
        )

        # Bidirectional row
        self._c_bidir = QCheckBox("Bidirectional")
        self._c_bidir.setStyleSheet("font-weight: normal;")
        self._c_bidir.setToolTip("Enable bidirectional scan correction")
        self._c_bidir_shift = QLineEdit("0.0")
        # self._c_bidir_shift.setMaximumWidth(65)
        self._c_bidir_shift.setValidator(QDoubleValidator(-0.5, 0.5, 6))
        self._c_bidir_shift.setToolTip(
            "Phase shift for bidirectional correction (pixels, −0.2 … 0.2)"
        )
        bidir_lbl = QLabel("Phase shift:")
        bidir_lbl.setStyleSheet("font-weight: normal;")
        # The label follows its field, so an unticked mode greys out as a
        # whole — the same cue a checkable section gives its contents.
        _follow_checkbox(self._c_bidir, self._c_bidir_shift, bidir_lbl)
        cg.addWidget(self._c_bidir, 3, 0, 1, 2)
        cg.addWidget(bidir_lbl, 3, 2)
        cg.addWidget(self._c_bidir_shift, 3, 3, 1, 3)

        # Harmonic (resonant) scan row — same pattern as bidirectional above.
        self._c_harmonic = QCheckBox("Harmonic")
        self._c_harmonic.setStyleSheet("font-weight: normal;")
        self._c_harmonic.setToolTip("Enable harmonic (resonant) scan handling")
        self._c_laser_duty = QLineEdit(str(DEFAULT_LASER_DUTY))
        self._c_laser_duty.setValidator(QDoubleValidator(0.0, 1.0, 4))
        self._c_laser_duty.setToolTip(
            "Fraction of the line period the laser is on"
        )
        self._c_line_start_delay = QLineEdit("0.0")
        self._c_line_stop_delay = QLineEdit("0.0")
        for edit, edge in (
            (self._c_line_start_delay, "start"),
            (self._c_line_stop_delay, "stop"),
        ):
            edit.setValidator(QDoubleValidator(-0.5, 0.5, 6))
            edit.setToolTip(
                f"Shift the {edge} edge of each reconstructed line "
                "(fraction of a line duration)"
            )
        duty_lbl = QLabel("Laser duty:")
        duty_lbl.setStyleSheet("font-weight: normal;")
        delay_lbl = QLabel("Line Δ start/stop:")
        delay_lbl.setStyleSheet("font-weight: normal;")
        _follow_checkbox(
            self._c_harmonic,
            duty_lbl,
            self._c_laser_duty,
            delay_lbl,
            self._c_line_start_delay,
            self._c_line_stop_delay,
        )
        cg.addWidget(self._c_harmonic, 4, 0)
        cg.addWidget(duty_lbl, 4, 1)
        cg.addWidget(self._c_laser_duty, 4, 2)
        cg.addWidget(delay_lbl, 4, 3)
        cg.addWidget(self._c_line_start_delay, 4, 4)
        cg.addWidget(self._c_line_stop_delay, 4, 5)

        self._cal_factor = QLineEdit("1+0j")
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

        self._c_chunk = QSpinBox()
        self._c_chunk.setRange(MIN_CHUNK_SIZE, 2_147_483_647)
        self._c_chunk.setSingleStep(100_000)
        self._c_chunk.setGroupSeparatorShown(True)
        self._c_chunk.setValue(DEFAULT_CHUNK_SIZE)
        self._c_chunk.setToolTip(
            "TTTR records read per iteration "
            f"(default {DEFAULT_CHUNK_SIZE:,}).\n"
            "Smaller = less memory and finer progress steps; "
            "larger = fewer iterations.\n"
        )

        cg.addWidget(QLabel("TCSPC bins:"), 2, 0)
        cg.addWidget(self._c_tcspc, 2, 1)
        cg.addWidget(QLabel("Calibration:"), 2, 2)
        cg.addWidget(self._cal_factor, 2, 3, 1, 3)

        cg.addWidget(QLabel("Chunk size:"), 5, 0)
        cg.addWidget(self._c_chunk, 5, 1, 1, 3)

        cfg_vlay.addLayout(cg)

        # ── action buttons, in a row under the fields ────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._populate_btn = QPushButton("Populate cfg")
        self._populate_btn.setToolTip(
            "Read the header of the first .ptu from input and fill in "
            "what it reports. Missing values are left as they are."
        )
        self._populate_btn.setEnabled(False)  # needs a PTU folder first
        btn_row.addWidget(self._populate_btn)
        btn_row.addWidget(_vsep())

        self._load_scan_btn = QPushButton("↓ Scan cfg")
        self._load_scan_btn.setToolTip(
            "Copy the scan config currently set in the File tab"
        )
        self._load_cal_btn = QPushButton("↓ Calibration")
        self._load_cal_btn.setToolTip(
            "Copy the calibration factor set in the Phasor tab"
        )
        for b in (self._load_scan_btn, self._load_cal_btn):
            b.setEnabled(False)
            btn_row.addWidget(b)

        btn_row.addWidget(_vsep())

        self._load_cfg_btn = QPushButton("Load cfg...")
        self._save_cfg_btn = QPushButton("Save cfg...")
        for b in (self._load_cfg_btn, self._save_cfg_btn):
            btn_row.addWidget(b)

        btn_row.addStretch()
        cfg_vlay.addLayout(btn_row)
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
        self._per_file_chk = QCheckBox("Per file table")
        self._single_tbl_chk = QCheckBox("Single merged table")
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
        run_box = QGroupBox()
        apply_style(run_box, S.GROUP_PLAIN)
        run_row = QHBoxLayout(run_box)
        run_row.setContentsMargins(3, 3, 3, 3)
        # run_row.setSpacing(6)
        self._run_btn = QPushButton("Run Batch")
        self._run_btn.setStyleSheet(
            S.BTN_RUN
            # + "QPushButton { min-width: 100px; }"
        )
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(S.BTN_STOP)
        run_row.addWidget(self._run_btn, 1)
        run_row.addSpacing(10)
        run_row.addWidget(self._stop_btn, 1)
        run_row.addStretch()
        root.addWidget(run_box)

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
        self._populate_btn.clicked.connect(self._on_populate_config)
        # The ↓ buttons need a PTU loaded in the File tab.
        self.state.file_loaded.connect(self._update_copy_enabled)
        self.state.dataset_changed.connect(self._on_dataset_changed)
        for edit in (self._c_frames, self._c_seqs, self._c_maxdet):
            edit.textChanged.connect(self._sync_slice_limits)
        self._sync_slice_limits()
        self._update_input_label()

    # ── directory pickers ────────────────────────────────────────────────

    def _sync_slice_limits(self):
        """Push the scan config's extents into the single-slice pickers."""

        def _n(edit, default: int) -> int:
            try:
                return max(1, int(edit.text().strip() or default))
            except ValueError:
                return default

        self._images_sec.set_dims(
            _n(self._c_frames, 1),
            _n(self._c_seqs, 1),
            _n(self._c_maxdet, 1),
        )

    def _browse_ptu(self):
        """Pick a folder — every .ptu inside it is processed."""
        d = QFileDialog.getExistingDirectory(
            self, "Select PTU folder", str(self._ptu_dir or "")
        )
        if not d:
            return
        self._ptu_dir = Path(d)
        self._selected_files = []  # a folder replaces any earlier file pick
        self._update_input_label()

    def _browse_ptu_files(self):
        """Pick individual .ptu files instead of a whole folder."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PTU files",
            str(self._ptu_dir or ""),
            "PTU files (*.ptu)",
        )
        if not paths:
            return
        self._selected_files = [Path(p) for p in paths]
        self._ptu_dir = self._selected_files[0].parent
        self._update_input_label()

    def _folder_files(self) -> list[Path]:
        """Every .ptu in the chosen folder, honouring 'Include sub-folders'."""
        if self._ptu_dir is None or not self._ptu_dir.is_dir():
            return []
        glob = "**/*.ptu" if self._recursive_chk.isChecked() else "*.ptu"
        return sorted(self._ptu_dir.glob(glob))

    def _update_input_label(self):
        """Report what is currently selected, and what that implies."""
        if self._selected_files:
            n = len(self._selected_files)
            self._input_lbl.setText(
                f"{n} file(s) selected in {self._ptu_dir.name}"
            )
            self._input_lbl.setToolTip(
                "\n".join(str(p) for p in self._selected_files)
            )
        elif self._ptu_dir is not None:
            found = len(self._folder_files())
            self._input_lbl.setText(
                f"Folder: {self._ptu_dir.name} — {found} .ptu found"
            )
            self._input_lbl.setToolTip(str(self._ptu_dir))
        else:
            self._input_lbl.setText("Nothing selected")
            self._input_lbl.setToolTip("")
        self._recursive_chk.setEnabled(not self._selected_files)
        self._update_populate_enabled()

    def _browse_lbl(self):
        d = QFileDialog.getExistingDirectory(self, "Select Labels folder")
        if d:
            self._lbl_edit.setText(d)

    # ── Config JSON ──────────────────────────────────────────────────────

    def _accum_list(self, n_seqs: int, *, strict: bool) -> list[int]:
        """Parse Accum/seq into one int per sequence.

        A single value is replicated across all sequences. With *strict* a
        length that is neither 1 nor *n_seqs* raises (used at run time); saving
        is lenient so a half-edited field can still be written out.
        """
        parts = [
            p.strip() for p in self._c_accu.text().split(",") if p.strip()
        ]
        if not parts:
            parts = ["1"]
        try:
            vals = [int(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"Accum/seq: {exc}") from exc
        if len(vals) == 1:
            return vals * max(1, n_seqs)
        if len(vals) == n_seqs or not strict:
            return vals
        raise ValueError(
            f"Accum/seq: got {len(vals)} values but N seqs = {n_seqs}. "
            f"Provide 1 value (applied to all) or exactly {n_seqs} values."
        )

    def current_settings(self, *, strict: bool = False) -> ScanSettings:
        """Read every scan widget — the panel's single source of truth.

        The JSON config, the tttrkit ScanConfig and the dataset attribute are
        all derived from this. With *strict* the accumulation count must match
        the sequence count, which is required before a run but not for saving
        a half-edited config.
        """

        def _num(edit, cast, default):
            txt = edit.text().strip()
            try:
                return cast(txt) if txt else default
            except ValueError:
                return default

        n_seqs = _num(self._c_seqs, int, 1)
        return ScanSettings(
            frames=_num(self._c_frames, int, 1),
            lines=_num(self._c_lines, int, 256),
            pixels=_num(self._c_pixels, int, 256),
            accumulations=tuple(self._accum_list(n_seqs, strict=strict)),
            max_detector=_num(self._c_maxdet, int, 4),
            bidirectional=self._c_bidir.isChecked(),
            bidirectional_phase_shift=_num(self._c_bidir_shift, float, 0.0),
            harmonic_scan=self._c_harmonic.isChecked(),
            laser_duty=_num(self._c_laser_duty, float, DEFAULT_LASER_DUTY),
            line_start_marker_delay=_num(self._c_line_start_delay, float, 0.0),
            line_stop_marker_delay=_num(self._c_line_stop_delay, float, 0.0),
            tcspc_bins=_num(self._c_tcspc, int, 4096),
            calib_factor=self._cal_factor.text().strip() or "1+0j",
        )

    def _config_dict(self) -> dict:
        """Scan config + calibration in the shared core JSON schema.

        Same schema the File tab reads and writes, so configs move between the
        two tabs unchanged. Chunk size is deliberately not stored — it is a
        machine-local speed knob, not part of the scan description.
        """
        return self.current_settings().to_json_dict()

    def _apply_dict(self, cfg: dict):
        """Push a config dict back into the UI.

        Reads the core schema (``accumulations``: list of ints, one per
        sequence) and still accepts the legacy batch-only ``accum_per_seq``
        string from configs saved by earlier versions of this panel.
        """
        s = cfg.get("scan", {})
        for key, edit in (
            ("frames", self._c_frames),
            ("lines", self._c_lines),
            ("pixels", self._c_pixels),
            ("max_detector", self._c_maxdet),
            ("tcspc_bins", self._c_tcspc),
        ):
            if s.get(key):
                edit.setText(str(int(s[key])))

        accum = s.get("accumulations", s.get("accum_per_seq"))
        seqs = s.get("sequences")
        if isinstance(accum, (list, tuple)):
            if seqs is None:
                seqs = len(accum)
            self._c_accu.setText(",".join(str(int(a)) for a in accum))
        elif accum is not None:
            self._c_accu.setText(str(accum))
        if seqs is not None:
            self._c_seqs.setText(str(seqs))

        if "bidirectional" in s:
            self._c_bidir.setChecked(bool(s["bidirectional"]))
        if "harmonic_scan" in s:
            self._c_harmonic.setChecked(bool(s["harmonic_scan"]))
        for key, edit in (
            ("bidirectional_phase_shift", self._c_bidir_shift),
            ("laser_duty", self._c_laser_duty),
            ("line_start_marker_delay", self._c_line_start_delay),
            ("line_stop_marker_delay", self._c_line_stop_delay),
        ):
            if key in s:
                edit.setText(str(s[key]))

        c = cfg.get("calibration") or {}
        if c.get("factor"):
            self._cal_factor.setText(str(c["factor"]))

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
        """Build a ScanConfig from the UI fields, validating accumulations."""
        return self.current_settings(strict=True).to_scan_config()

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
        if self._ptu_dir is None or not self._ptu_dir.is_dir():
            self._log_line("Select a PTU folder or files first.", error=True)
            return
        ptu_dir = str(self._ptu_dir)

        img_cfgs = self._images_sec.get_configs()
        phasor_cfgs = self._phasor_sec.get_configs()
        decay_cfgs = self._decay_sec.get_configs()
        if not any([img_cfgs, phasor_cfgs, decay_cfgs]):
            self._log_line("Enable at least one export section.", error=True)
            return

        try:
            settings = self.current_settings(strict=True)
            scan_cfg = settings.to_scan_config()
            cal = self._cal_complex()
        except Exception as e:
            self._log_line(f"Config error: {e}", error=True)
            return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(ptu_dir) / f"batch_{ts}"
        self._out_dir_label.setText(f"Output: {out_dir}")

        params = dict(
            ptu_dir=ptu_dir,
            ptu_files=[str(p) for p in self._selected_files] or None,
            lbl_dir=self._lbl_edit.text().strip() or None,
            recursive=self._recursive_chk.isChecked(),
            scan_config=scan_cfg,
            scan_config_dict=settings.to_json_dict()["scan"],
            tcspc_bins=int(self._c_tcspc.text() or 0) or None,
            chunk_size=self._c_chunk.value(),
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

    def _update_copy_enabled(self):
        """The ↓ buttons copy the File tab's fields — useless before a load."""
        available = self.state.file_config() is not None
        self._load_scan_btn.setEnabled(available)
        self._load_cal_btn.setEnabled(available)

    def _on_dataset_changed(self):
        self._update_copy_enabled()

    def _first_ptu(self) -> Path | None:
        """First file the run will process, whichever way the input was set."""
        files = self._selected_files or self._folder_files()
        return files[0] if files else None

    def _update_populate_enabled(self):
        self._populate_btn.setEnabled(self._first_ptu() is not None)

    def _on_populate_config(self):
        """Fill the fields from the first PTU's header — the batch equivalent
        of 'Read PTU…' in the File tab.

        Only what the header actually reports is written; anything it does not
        carry (sequences, accumulations, max detector, bidirectional) keeps its
        current value, and the log says which values came from the file.
        """
        from napari_flopa.core.io.loader import read_ptu_file

        path = self._first_ptu()
        if path is None:
            self._log_line(
                "No .ptu file found in the selected folder.", error=True
            )
            return
        try:
            ptu_data = read_ptu_file(str(path), header=False)
        except Exception as e:
            self._log_line(f"Could not read {path.name}: {e}", error=True)
            return

        tags = ptu_data["header"]
        constants = ptu_data["constants"]
        csrc = ptu_data.get("constants_source", {})
        applied: list[str] = []

        px_x = tags.get("ImgHdr_PixX")
        px_y = tags.get("ImgHdr_PixY")
        n_frames = tags.get("ImgHdr_NumberOfFrames")
        if isinstance(px_x, (int, float)):
            self._c_pixels.setText(str(int(px_x)))
            applied.append(f"pixels={int(px_x)}")
        if isinstance(px_y, (int, float)):
            self._c_lines.setText(str(int(px_y)))
            applied.append(f"lines={int(px_y)}")
        if isinstance(n_frames, (int, float)) and n_frames > 0:
            self._c_frames.setText(str(int(n_frames)))
            applied.append(f"frames={int(n_frames)}")

        bins = constants.get("tcspc_bins")
        if bins:
            self._c_tcspc.setText(str(int(bins)))
            applied.append(
                f"tcspc_bins={int(bins)} "
                f"({csrc.get('tcspc_bins', 'default')})"
            )
        rep = constants.get("repetition_rate")
        if rep:
            applied.append(f"rep. rate={float(rep) / 1e6:.6g} MHz (header)")

        self._log_line(f"Populated from {path.name}: " + ", ".join(applied))
        self._log_line(
            "Sequences, accum/seq, max detector and bidirectional are not in "
            "the header — check them manually."
        )

    def _on_load_scan(self):
        """Copy the scan config the File tab currently shows.

        Those are the tunable fields exposed once a PTU is selected there —
        reconstruction is not required. The button is disabled until then.
        """
        cfg = self.state.file_config()
        if cfg is None:
            self._log_line(
                "Load a PTU file in the File tab first.", error=True
            )
            return
        self._apply_dict({"scan": cfg.get("scan", {})})
        self._log_line("Scan config copied from the File tab.")

    def _on_load_calibration(self):
        """Copy the calibration factor set in the Phasor tab."""
        factor = self.state.calib_factor
        self._cal_factor.setText(
            f"{factor.real:.6g}+{factor.imag:.6g}j"
            if factor.imag >= 0
            else f"{factor.real:.6g}{factor.imag:.6g}j"
        )
        self._log_line(f"Calibration copied: factor {factor}")

    def _log_line(self, msg: str, *, error: bool = False):
        self._log.appendHtml((_ERR if error else _INFO).format(msg))
