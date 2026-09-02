"""
TCSPC decay curve panel.

Dataset variable used: tcspc_histogram  dims: (frame, channel, tcspc_channel).
The sequence dimension is not stored in the TCSPC histogram; the Sequences
aggregation checkbox is accepted but is a no-op unless future reconstructions
add that dimension.

Default state (no aggregation): plots one curve per (frame × channel) combination,
auto-coloured from a 60-entry palette built from matplotlib tab20/tab20b/tab20c.

Aggregation behaviour:
  Frames only    → one curve per detector channel
  Detectors only → one curve per frame
  Both           → single summed curve
  Sequences      → no-op (no sequence dim in tcspc_histogram)

Per-detector toggle buttons (D0, D1, …) show/hide all curves for that detector;
disabled when "Detectors" is aggregated.  All / None buttons are always active.

Matplotlib legend is capped at _MAX_LEGEND (8) entries.
The scrollable curve list uses rows of QHBoxLayouts for tight top-left alignment.

Export (selector + Save, same pattern as the Phasor tab):
  • Plot  → PNG / SVG / PDF
  • Table → CSV: time_ns + one column per curve (every curve, hidden ones
            included), with shift and normalisation applied as plotted
"""

import csv
import itertools
import traceback
from pathlib import Path

import numpy as np
from matplotlib import colormaps as _mpl_colormaps
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
    )
except ImportError:
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
    )

from tttrkit.analysis.decay import shift_decay

from napari_flopa.ui.state import FlopaState
from napari_flopa.ui.style import MPL, S, apply_style
from napari_flopa.ui.widgets.status_label import StatusLabel


# 60-colour palette from matplotlib's three tab20 maps
def _build_palette() -> list:
    colours = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = _mpl_colormaps[cmap_name]
        for i in range(20):
            r, g, b, _ = cmap(i / 20)
            colours.append(
                f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            )
    return colours


_PALETTE = _build_palette()

_LEGEND_COLS = 5  # columns in the scrollable curve grid

_MAX_LEGEND = 8  # max entries shown in the matplotlib plot legend


class DecayPanel(QWidget):
    """
    Decay tab — TCSPC decay curves.

    Reads tcspc_histogram from FlopaState.dataset and plots one curve per free
    dimension combination after optional aggregation.  Receives view settings
    from FlimViewPanel via on_view_changed(dict) and marks itself stale when
    settings differ from the last plot.
    """

    def __init__(self, state: FlopaState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer

        self._current_view: dict = {}
        self._plotted_view: dict | None = None

        # List of dicts: {label, time_ns, counts, color, visible, chan_idx}
        self._curves: list = []
        # Det-button widgets keyed by channel index
        self._det_btns: dict[int, QPushButton] = {}

        self._build_ui()
        self.state.dataset_changed.connect(self._on_dataset_changed)

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Top toolbar ────────────────────────────────────────────────
        tb = QHBoxLayout()
        tb.setSpacing(4)

        self._plot_btn = QPushButton("Plot")
        self._plot_btn.setStyleSheet(S.BTN_RUN)
        self._plot_btn.setEnabled(False)
        self._plot_btn.setToolTip(
            "Plot decay curves from the current dataset / aggregation settings"
        )
        tb.addWidget(self._plot_btn)

        self._stale = QLabel("●")
        self._stale.setFixedWidth(14)
        self._stale.setAlignment(Qt.AlignCenter)
        self._stale.setStyleSheet(S.STALE_INACTIVE)
        self._stale.setVisible(False)
        tb.addWidget(self._stale)

        self._from_view_btn = QPushButton("FLIM View")
        self._from_view_btn.setEnabled(False)
        self._from_view_btn.setToolTip(
            "Settings from FLIM View (only current selection)"
        )
        tb.addWidget(self._from_view_btn)

        self._default_btn = QPushButton("Default")
        self._default_btn.setToolTip(
            "Reset to default: no aggregation, all curves visible"
        )
        tb.addWidget(self._default_btn)

        tb.addStretch()

        # Same pattern as the Phasor tab: pick what to save, then Save.
        self._export_combo = QComboBox()
        self._export_combo.addItem("Save plot…", "plot")
        self._export_combo.addItem("Save table…", "table")
        self._export_combo.setToolTip(
            "Plot: PNG / SVG / PDF of the figure\n"
            "Table: CSV with one column per curve (shift + normalisation "
            "applied, as plotted)"
        )
        tb.addWidget(self._export_combo)
        self._save_btn = QPushButton("Save")
        self._save_btn.setEnabled(False)
        tb.addWidget(self._save_btn)

        root.addLayout(tb)

        # ── Settings row ───────────────────────────────────────────────
        sr = QHBoxLayout()
        sr.setSpacing(6)

        agg_box = QGroupBox("Aggregate")
        apply_style(agg_box, S.GROUP_PRIMARY)
        agg_lay = QHBoxLayout(agg_box)
        agg_lay.setSpacing(6)
        agg_lay.setContentsMargins(6, 2, 6, 2)
        self._agg_frames = QCheckBox("Frames")
        self._agg_seqs = QCheckBox("Sequences")
        self._agg_chans = QCheckBox("Detectors")
        # Default: nothing aggregated → all frame×channel (×sequence if present) curves
        self._agg_frames.setChecked(False)
        self._agg_seqs.setChecked(False)
        self._agg_chans.setChecked(False)
        for w in (self._agg_frames, self._agg_seqs, self._agg_chans):
            agg_lay.addWidget(w)
        sr.addWidget(agg_box)

        shift_box = QGroupBox("Shift (bins)")
        apply_style(shift_box, S.GROUP_PRIMARY)
        shift_lay = QHBoxLayout(shift_box)
        shift_lay.setContentsMargins(6, 2, 6, 2)
        self._shift_spin = QSpinBox()
        self._shift_spin.setRange(-1000, 1000)
        self._shift_spin.setValue(0)
        self._shift_spin.setToolTip("Shift all curves by N bins.")
        shift_lay.addWidget(self._shift_spin)
        sr.addWidget(shift_box)

        # sr.addStretch()
        root.addLayout(sr)

        # ── Display options row (below Aggregate) ──────────────────────────────
        sr2 = QHBoxLayout()
        sr2.setSpacing(6)

        scale_box = QGroupBox("Display options")
        apply_style(scale_box, S.GROUP_PRIMARY)
        scale_lay = QHBoxLayout(scale_box)
        scale_lay.setSpacing(6)
        scale_lay.setContentsMargins(6, 2, 6, 2)
        self._log_check = QCheckBox("Log Y")
        self._log_check.setChecked(True)
        self._log_check.setToolTip("Logarithmic Y axis")
        self._norm_check = QCheckBox("Norm")
        self._norm_check.setToolTip("Normalize each curve to its peak")
        scale_lay.addWidget(self._log_check)
        scale_lay.addWidget(self._norm_check)
        sr2.addWidget(scale_box)

        root.addLayout(sr2)
        root.addStretch()
        # ── Matplotlib canvas ──────────────────────────────────────────
        # Small default size (keeps the initial dock narrow); the Expanding
        # size policy + widgetResizable scroll area stretch it to fill the tab.
        self._fig = Figure(figsize=(4.5, 4.5), tight_layout=True)
        self._fig.patch.set_facecolor(MPL.FIG_BG)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(MPL.AXES_BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self._canvas.setMaximumHeight(550)
        root.addWidget(self._canvas)
        self._draw_empty()
        self._canvas.setVisible(False)  # hidden until the first plot

        # ── Per-detector toggle buttons ────────────────────────────────
        self._det_row_widget = QWidget()
        det_row = QHBoxLayout(self._det_row_widget)
        det_row.setContentsMargins(0, 0, 0, 0)
        det_row.setSpacing(4)
        det_row.addWidget(QLabel("Detectors:"))
        self._det_btns_layout = QHBoxLayout()
        self._det_btns_layout.setSpacing(3)
        det_row.addLayout(self._det_btns_layout)
        det_row.addStretch()

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(S.SEPARATOR)
        det_row.addWidget(sep)

        self._show_all_btn = QPushButton("All")
        self._show_all_btn.setFixedWidth(36)
        self._show_all_btn.clicked.connect(lambda: self._set_all_visible(True))
        self._hide_all_btn = QPushButton("None")
        self._hide_all_btn.setFixedWidth(40)
        self._hide_all_btn.clicked.connect(
            lambda: self._set_all_visible(False)
        )
        det_row.addWidget(self._show_all_btn)
        det_row.addWidget(self._hide_all_btn)

        self._det_row_widget.setVisible(False)
        root.addWidget(self._det_row_widget)

        # ── Scrollable per-curve legend (rows of QHBoxLayouts, top-left) ──
        self._legend_scroll = QScrollArea()
        self._legend_scroll.setWidgetResizable(True)
        self._legend_scroll.setMaximumHeight(120)
        self._legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._legend_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._legend_scroll.setVisible(False)
        self._legend_inner = QWidget()
        self._legend_vbox = QVBoxLayout(self._legend_inner)
        self._legend_vbox.setSpacing(1)
        self._legend_vbox.setContentsMargins(2, 2, 2, 2)
        self._legend_vbox.addStretch()  # keeps rows pushed to top
        self._legend_scroll.setWidget(self._legend_inner)
        root.addWidget(self._legend_scroll)

        # ── Status ─────────────────────────────────────────────────────
        self._status = StatusLabel(
            "Load and reconstruct a PTU file to enable decay plotting."
        )
        root.addWidget(self._status)

        # ── Wiring ─────────────────────────────────────────────────────
        self._plot_btn.clicked.connect(self._on_plot)
        self._from_view_btn.clicked.connect(self._on_from_view)
        self._default_btn.clicked.connect(self._on_default)
        self._save_btn.clicked.connect(self._on_export)
        self._log_check.toggled.connect(self._redraw)
        self._norm_check.toggled.connect(self._redraw)
        self._shift_spin.valueChanged.connect(self._redraw)

        # Aggregation changes → mark stale
        for chk in (self._agg_frames, self._agg_seqs, self._agg_chans):
            chk.toggled.connect(self._mark_stale)
        # Also disable/enable det buttons when Detectors agg changes
        self._agg_chans.toggled.connect(self._update_det_btn_state)

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    @Slot()
    def _on_dataset_changed(self):
        """Enable/disable Plot and From View when a new dataset arrives; reset stale state."""
        ds = self.state.dataset
        has_tcspc = ds is not None and "tcspc_histogram" in ds
        self._plot_btn.setEnabled(has_tcspc)
        self._from_view_btn.setEnabled(has_tcspc)
        if not has_tcspc:
            self._status.info("No TCSPC data. Reconstruct with 'All' output.")
        else:
            self._status.info("TCSPC data ready. Click 'Plot'.")
        self._plotted_view = None
        self._stale.setVisible(False)
        self._curves = []
        self._canvas.setVisible(False)  # hide until (re)plotted

    @Slot(dict)
    def on_view_changed(self, settings: dict):
        """
        Receive view settings from FlimViewPanel.

        Stores the dict (frame, sequence, channel, sum_frames, sum_sequences,
        sum_channels) and marks the panel stale if curves are already plotted.
        """
        self._current_view = settings
        self._mark_stale()

    def _mark_stale(self):
        """Turn the stale indicator red if curves exist and settings have changed."""
        if self._curves:
            self._stale.setStyleSheet(S.STALE_STALE)
            self._stale.setToolTip(
                "Settings changed since last plot — click Plot to refresh."
            )
            self._stale.setVisible(True)

    def showEvent(self, event):
        """Re-apply tight_layout on first show to fix canvas size computed at zero size."""
        super().showEvent(event)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Plot / From View / Default actions                                   #
    # ------------------------------------------------------------------ #

    def _on_plot(self):
        """Extract all curves with current aggregation settings and refresh the plot."""
        curves = self._extract_all_curves()
        if not curves:
            return
        self._curves = curves
        self._plotted_view = dict(self._current_view)
        self._stale.setStyleSheet(S.STALE_FRESH)
        self._stale.setToolTip("Plot is up to date.")
        self._stale.setVisible(True)
        self._save_btn.setEnabled(True)
        self._rebuild_det_buttons()
        self._rebuild_legend()
        self._redraw()

    def _on_from_view(self):
        """Copy aggregation flags from FLIM View, then plot and hide non-matching curves."""
        v = self._current_view
        # Block signals so each checkbox change doesn't separately mark stale
        for chk, key in (
            (self._agg_frames, "sum_frames"),
            (self._agg_seqs, "sum_sequences"),
            (self._agg_chans, "sum_channels"),
        ):
            chk.blockSignals(True)
            chk.setChecked(v.get(key, False))
            chk.blockSignals(False)

        curves = self._extract_all_curves()
        if not curves:
            return

        # Auto-hide curves that don't match the current view selection
        # (only applies to non-aggregated dims)
        agg_frames = self._agg_frames.isChecked()
        agg_seqs = self._agg_seqs.isChecked()
        agg_chans = self._agg_chans.isChecked()
        sel_frame = v.get("frame", 0)
        sel_seq = v.get("sequence", 0)
        sel_chan = v.get("channel", 0)

        # Parse indices from label tokens like "F2", "S1", "D0"
        def _idx(prefix, label):
            for tok in label.split():
                if tok.startswith(prefix) and tok[len(prefix) :].isdigit():
                    return int(tok[len(prefix) :])
            return None

        for c in curves:
            label = c["label"]
            show = True
            if not agg_frames:
                fi = _idx("F", label)
                if fi is not None and fi != sel_frame:
                    show = False
            if show and not agg_seqs:
                si = _idx("S", label)
                if si is not None and si != sel_seq:
                    show = False
            if show and not agg_chans:
                di = _idx("D", label)
                if di is not None and di != sel_chan:
                    show = False
            c["visible"] = show

        self._curves = curves
        self._plotted_view = dict(v)
        self._stale.setStyleSheet(S.STALE_FRESH)
        self._stale.setToolTip("Plot is up to date.")
        self._stale.setVisible(True)
        self._save_btn.setEnabled(True)
        self._rebuild_det_buttons()
        self._rebuild_legend()
        self._redraw()

    def _on_default(self):
        """Reset to no aggregation, all curves visible, and re-plot.

        Clearing the aggregation flags changes which curves exist, so this
        re-extracts and redraws immediately (like From View) rather than
        leaving the old curves on screen behind a stale dot.
        """
        for chk in (self._agg_frames, self._agg_seqs, self._agg_chans):
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        self._on_plot()  # no-op if there is no data to extract
        for c in self._curves:
            c["visible"] = True
        self._rebuild_det_buttons()
        self._rebuild_legend()
        self._redraw()

    # ------------------------------------------------------------------ #
    # Data extraction                                                      #
    # ------------------------------------------------------------------ #

    def _extract_all_curves(self) -> list:
        """
        Build a list of curve dicts from tcspc_histogram given current aggregation.

        Applies sum over any checked aggregation dims (frame / sequence / channel),
        then takes the cartesian product of all remaining free dims to produce one
        curve per combination.  Each dict contains:
          label    — e.g. "F0 D1"
          time_ns  — 1-D array of bin centres in nanoseconds
          counts   — 1-D float64 photon counts
          color    — hex colour string from _PALETTE
          visible  — True (all visible by default)
          chan_idx — int channel index, or None when channel was aggregated
        Returns an empty list on error (message written to status bar).
        """
        ds = self.state.dataset
        if ds is None or "tcspc_histogram" not in ds:
            return []
        try:
            da = ds["tcspc_histogram"]
            # Expected dims: (frame, channel, tcspc_channel)

            agg_frames = self._agg_frames.isChecked()
            agg_seqs = self._agg_seqs.isChecked()
            agg_chans = self._agg_chans.isChecked()

            if agg_frames and "frame" in da.dims:
                da = da.sum("frame")
            if agg_seqs and "sequence" in da.dims:
                da = da.sum("sequence")
            if agg_chans and "channel" in da.dims:
                da = da.sum("channel")

            time_dim = "tcspc_channel"
            free_dims = [d for d in da.dims if d != time_dim]

            ip = ds.attrs.get("instrument_params", {})
            res_ns = float(ip.get("tcspc_resolution_ns", 1.0))
            n_bins = da.sizes[time_dim]
            time_ns = np.arange(n_bins) * res_ns

            _short = {"frame": "F", "sequence": "S", "channel": "D"}

            curves = []
            if not free_dims:
                counts = da.values.flatten().astype(np.float64)
                curves.append(
                    {
                        "label": "Sum",
                        "time_ns": time_ns,
                        "counts": counts,
                        "color": _PALETTE[0],
                        "visible": True,
                        "chan_idx": None,
                    }
                )
            else:
                sizes = [da.sizes[d] for d in free_dims]
                for combo in itertools.product(*[range(s) for s in sizes]):
                    sel = {
                        d: int(v)
                        for d, v in zip(free_dims, combo, strict=True)
                    }
                    counts = da.isel(**sel).values.flatten().astype(np.float64)
                    label = " ".join(
                        f"{_short.get(d, d[0].upper())}{v}"
                        for d, v in sel.items()
                    )
                    chan_idx = sel.get("channel")
                    curves.append(
                        {
                            "label": label,
                            "time_ns": time_ns.copy(),
                            "counts": counts,
                            "color": _PALETTE[len(curves) % len(_PALETTE)],
                            "visible": True,
                            "chan_idx": chan_idx,
                        }
                    )

            return curves

        except Exception as e:
            traceback.print_exc()
            self._status.error(f"Error extracting curves: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Drawing                                                              #
    # ------------------------------------------------------------------ #

    def _redraw(self):
        """
        Re-render the matplotlib axes from self._curves.

        Applies shift (via shift_decay) and optional normalisation at draw time
        without mutating stored counts.  Limits the in-plot legend to _MAX_LEGEND
        entries,
        appending a "… +N more" label when over the cap.  Y floor: 1.0 for log,
        1e-4 for normalised log.
        """
        # Show the canvas only once there are curves to draw (matches phasor).
        self._canvas.setVisible(bool(self._curves))
        if not self._curves:
            self._draw_empty()
            return

        shift = self._shift_spin.value()
        log = self._log_check.isChecked()
        norm = self._norm_check.isChecked()

        ax = self._ax
        ax.clear()
        ax.set_facecolor(MPL.AXES_BG)

        # Subtle grid
        ax.grid(True, color=MPL.GRID, linewidth=0.5, linestyle="--", alpha=0.9)
        ax.set_axisbelow(True)

        visible = [c for c in self._curves if c["visible"]]
        for curve in visible:
            counts = shift_decay(curve["counts"], -shift)
            if norm:
                peak = counts.max()
                counts = counts / peak if peak > 0 else counts
            ax.step(
                curve["time_ns"],
                np.maximum(counts, 1e-10 if log else 0),
                where="mid",
                color=curve["color"],
                linewidth=0.9,
                label=curve["label"],
            )

        ax.set_yscale("log" if log else "linear")
        if log:
            ax.set_ylim(bottom=1e-4 if norm else 1.0)

        res_ns = 1.0
        if self._curves:
            dt = np.diff(self._curves[0]["time_ns"])
            res_ns = float(dt[0]) if len(dt) > 0 else 1.0
        ax.set_xlabel(
            "Time (ns)" if res_ns != 1.0 else "TCSPC bin",
            color=MPL.TICK,
            fontsize=8,
        )
        ax.set_ylabel(
            "Counts" if not norm else "Norm. counts",
            color=MPL.TICK,
            fontsize=8,
        )
        # which="both": on a log axis the minor ticks/labels keep matplotlib's
        # default black unless they are styled explicitly.
        ax.tick_params(which="both", colors=MPL.TICK, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(MPL.SPINE)

        if visible:
            ax.set_xlim(visible[0]["time_ns"][0], visible[0]["time_ns"][-1])

        # Legend — cap at _MAX_LEGEND
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            n_over = max(0, len(handles) - _MAX_LEGEND)
            h_show = handles[:_MAX_LEGEND]
            l_show = labels[:_MAX_LEGEND]
            if n_over:
                l_show[-1] = f"… +{n_over + 1} more"
            ax.legend(
                h_show,
                l_show,
                fontsize=7,
                loc="upper right",
                facecolor=MPL.AXES_BG,
                edgecolor=MPL.SPINE,
                labelcolor=MPL.TICK,
                framealpha=0.8,
            )

        self._canvas.draw_idle()

        n_vis = len(visible)
        n_tot = len(self._curves)
        peak_str = ""
        if n_vis == 1:
            counts = np.roll(visible[0]["counts"], shift)
            peak_str = f"  Peak: {int(counts.max()):,}"
        self._status.info(
            f"{n_vis}/{n_tot} curve(s) visible, shift={shift} bins.{peak_str}"
        )

    def _draw_empty(self):
        """Render a placeholder "No data" axes with consistent styling."""
        ax = self._ax
        ax.clear()
        ax.set_facecolor(MPL.AXES_BG)
        ax.grid(True, color=MPL.GRID, linewidth=0.5, linestyle="--", alpha=0.6)
        ax.set_xlabel("Time (ns)", color=MPL.TICK, fontsize=8)
        ax.set_ylabel("Counts", color=MPL.TICK, fontsize=8)
        # which="both": on a log axis the minor ticks/labels keep matplotlib's
        # default black unless they are styled explicitly.
        ax.tick_params(which="both", colors=MPL.TICK, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(MPL.SPINE)
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            color="#555555",
            fontsize=11,
            transform=ax.transAxes,
        )
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Per-detector buttons                                                 #
    # ------------------------------------------------------------------ #

    def _rebuild_det_buttons(self):
        """
        Recreate per-detector toggle buttons (D0, D1, …) from current curves.

        The detector row is visible whenever curves exist; D-buttons are disabled
        when "Detectors" aggregation is checked.  All / None buttons are unaffected.
        """
        # Remove old D-buttons (keep the All/None buttons in the row)
        for btn in self._det_btns.values():
            self._det_btns_layout.removeWidget(btn)
            btn.deleteLater()
        self._det_btns.clear()

        # Always show the row (All/None are always useful)
        self._det_row_widget.setVisible(bool(self._curves))

        chan_indices = sorted(
            {c["chan_idx"] for c in self._curves if c["chan_idx"] is not None}
        )
        disabled = self._agg_chans.isChecked()

        for ch in chan_indices:
            btn = QPushButton(f"D{ch}")
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedWidth(32)
            btn.setEnabled(not disabled)
            btn.setStyleSheet(
                S.BTN_DET_ON if not disabled else S.BTN_DET_DISABLED
            )
            btn.toggled.connect(
                lambda checked, d=ch: self._on_det_toggled(d, checked)
            )
            self._det_btns_layout.addWidget(btn)
            self._det_btns[ch] = btn

    def _update_det_btn_state(self, detectors_aggregated: bool):
        """Enable/disable D-buttons when the Detectors aggregation checkbox changes."""
        for btn in self._det_btns.values():
            btn.setEnabled(not detectors_aggregated)
            btn.setStyleSheet(
                S.BTN_DET_DISABLED
                if detectors_aggregated
                else (S.BTN_DET_ON if btn.isChecked() else S.BTN_DET_OFF)
            )

    def _on_det_toggled(self, chan_idx: int, visible: bool):
        """Show/hide all curves belonging to detector chan_idx and sync button style."""
        btn = self._det_btns.get(chan_idx)
        if btn:
            btn.setStyleSheet(S.BTN_DET_ON if visible else S.BTN_DET_OFF)
        for c in self._curves:
            if c["chan_idx"] == chan_idx:
                c["visible"] = visible
        self._rebuild_legend()
        self._redraw()

    # ------------------------------------------------------------------ #
    # Scrollable per-curve legend                                          #
    # ------------------------------------------------------------------ #

    def _rebuild_legend(self):
        """
        Rebuild the scrollable curve list.

        Lays out _CurveLegendEntry widgets in rows of _LEGEND_COLS using
        QHBoxLayouts inside the vertical scroll area.  A trailing stretch in each
        row and at the bottom of the VBox keeps entries packed to the top-left.
        """
        # Clear all rows (keep the trailing stretch at index 0)
        while self._legend_vbox.count() > 1:
            item = self._legend_vbox.takeAt(0)
            if item.layout():
                while item.layout().count():
                    w = item.layout().takeAt(0)
                    if w.widget():
                        w.widget().deleteLater()

        if not self._curves:
            self._legend_scroll.setVisible(False)
            return

        self._legend_scroll.setVisible(True)
        rows = [
            self._curves[i : i + _LEGEND_COLS]
            for i in range(0, len(self._curves), _LEGEND_COLS)
        ]
        base = 0
        for row_curves in rows:
            row_lay = QHBoxLayout()
            row_lay.setSpacing(2)
            row_lay.setContentsMargins(0, 0, 0, 0)
            for j, curve in enumerate(row_curves):
                k = base + j
                entry = _CurveLegendEntry(
                    curve["label"], curve["color"], curve["visible"]
                )
                entry.toggled.connect(
                    lambda checked, idx=k: self._on_curve_toggled(idx, checked)
                )
                row_lay.addWidget(entry)
            row_lay.addStretch()
            self._legend_vbox.insertLayout(
                self._legend_vbox.count() - 1, row_lay
            )
            base += len(row_curves)

    def _on_curve_toggled(self, idx: int, visible: bool):
        """Toggle visibility of one curve and sync its detector button if applicable."""
        if 0 <= idx < len(self._curves):
            self._curves[idx]["visible"] = visible
            # Sync det button state: if any curve for that det is visible, btn=on
            chan = self._curves[idx]["chan_idx"]
            if chan is not None and chan in self._det_btns:
                any_vis = any(
                    c["visible"] for c in self._curves if c["chan_idx"] == chan
                )
                btn = self._det_btns[chan]
                btn.blockSignals(True)
                btn.setChecked(any_vis)
                btn.setStyleSheet(S.BTN_DET_ON if any_vis else S.BTN_DET_OFF)
                btn.blockSignals(False)
            self._redraw()

    def _set_all_visible(self, visible: bool):
        """Show or hide all curves at once; syncs all detector button states."""
        for c in self._curves:
            c["visible"] = visible
        # Sync detector buttons
        for _ch, btn in self._det_btns.items():
            btn.blockSignals(True)
            btn.setChecked(visible)
            btn.setStyleSheet(S.BTN_DET_ON if visible else S.BTN_DET_OFF)
            btn.blockSignals(False)
        self._rebuild_legend()
        self._redraw()

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def _on_export(self):
        """Dispatch on the export selector (plot | table)."""
        if not self._curves:
            return
        if self._export_combo.currentData() == "plot":
            self._save_plot()
        else:
            self._save_table()

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Decay Plot",
            "decay.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        try:
            self._fig.savefig(
                path, dpi=300, facecolor=MPL.FIG_BG, bbox_inches="tight"
            )
            self._status.info(f"Plot saved: {Path(path).name}")
        except Exception as e:
            self._status.error(f"Save error: {e}")

    def _save_table(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Decay Table", "decay.csv", "CSV (*.csv)"
        )
        if not path:
            return
        self._export_csv(path)

    def _export_csv(self, path: str):
        """
        Write all curves to a CSV file.

        Columns: time_ns, then one column per curve (label as header).  Shift and
        normalisation are applied; shorter curves are zero-padded to the longest.
        """
        try:
            shift = self._shift_spin.value()
            norm = self._norm_check.isChecked()
            max_len = max(len(c["time_ns"]) for c in self._curves)
            ref_time = max(self._curves, key=lambda c: len(c["time_ns"]))[
                "time_ns"
            ]

            cols, headers = [], ["time_ns"]
            for c in self._curves:
                # Match the plot exactly: _redraw uses shift_decay(counts, -shift).
                # (The old np.roll(counts, +shift) shifted the CSV the opposite way.)
                col = shift_decay(c["counts"], -shift)
                if norm:
                    peak = col.max()
                    col = col / peak if peak > 0 else col
                if len(col) < max_len:
                    col = np.pad(col, (0, max_len - len(col)))
                cols.append(col)
                headers.append(c["label"].replace(" ", "_"))

            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for i in range(max_len):
                    writer.writerow(
                        [f"{ref_time[i]:.6f}"]
                        + [f"{col[i]:.4f}" for col in cols]
                    )
            self._status.info(
                f"Table saved: {Path(path).name} "
                f"({len(self._curves)} curve(s), {max_len:,} bins)"
            )
        except Exception as e:
            traceback.print_exc()
            self._status.error(f"CSV error: {e}")


# ── Legend entry widget ───────────────────────────────────────────────────


class _CurveLegendEntry(QWidget):
    """
    Compact legend item: coloured square swatch + labelled checkbox.

    Emits toggled(bool) when the checkbox state changes.  Used inside the
    scrollable legend area of DecayPanel.
    """

    toggled = Signal(bool)

    def __init__(self, label: str, color: str, visible: bool, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 6, 1)
        lay.setSpacing(3)

        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(
            f"background-color: {color}; border: 1px solid #555; border-radius: 2px;"
        )
        lay.addWidget(swatch)

        chk = QCheckBox(label)
        chk.setChecked(visible)
        chk.toggled.connect(self.toggled)
        lay.addWidget(chk)
