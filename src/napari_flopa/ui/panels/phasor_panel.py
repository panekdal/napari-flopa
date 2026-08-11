"""
Phasor analysis panel — modelled on IMCF-Biocev/FLOPA.

Color / plot rules (matching old FLOPA behaviour):
  Per pixel, no mask   → sub-modes: Scatter (white) | Intensity Weighted (alpha) | Density
  Per pixel, mask      → forced "Labels" mode; colors from napari layer .get_color()
  Per pixel, mask + "As filter"
                       → mask treated as binary: only its pixels are plotted, as
                         one population, so the sub-modes apply again
  Per object, no mask  → single centroid (white); no background scatter
  Per object, mask     → one centroid per label; colors from napari layer; no background scatter
  Cmap selector        → enabled only for Density (no mask)

Calibration:
  • Factor stored as complex number; applied as phasor_complex * factor
  • "Calculate" button → CalibrationDialog: enter τ_ref (ns) + measured G, S → computes factor
  • "Revert"           → the factor that came with the loaded config (1+0j if none)
  • "1+0j"             → no calibration

Stale indicator (●):  turns red on ANY change to view selection OR plot settings after last plot.

Export:
  • Plot  → PNG / SVG / PDF
  • Table → CSV (pandas): dataset_name, label_id, g, s, photon_count_sum, area_pixels
            (populated after every plot, not only per-object; holds every valid
             point even when the scatter was subsampled for display)
"""

import contextlib
import traceback
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from qtpy.QtCore import Qt, Slot
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
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

from napari_flopa.ui.state import FlopaState
from napari_flopa.ui.style import MPL, S, apply_style
from napari_flopa.ui.widgets.status_label import StatusLabel

# Scatter subsample cap — display only; the CSV export always holds every
# valid pixel. 300k covers a full 500×500 frame without thinning.
_MAX_PX = 300_000


class PhasorPanel(QWidget):
    """Phasor tab — phasor analysis."""

    # Pixel plot sub-modes (only relevant for per-pixel, no mask)
    PM_SCATTER = 0
    PM_INTENSITY = 1  # alpha-weighted by photon count
    PM_DENSITY = 2  # 2D histogram imshow

    def __init__(self, state: FlopaState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer

        self._current_view: dict = {}
        self._plotted_settings: dict | None = None
        self._final_plot_data: dict | None = None  # points actually drawn
        self._export_data: dict | None = None  # every valid point, for CSV
        self._calib_factor: complex = 1.0 + 0j
        # Factor handed over by the last reconstruction (from the config file),
        # i.e. what "Revert" goes back to; 1+0j when the config carried none.
        self._loaded_calib_factor: complex = 1.0 + 0j

        # ROI selection state
        self._roi_active = False
        self._roi_label_id = 0
        self._roi_lasso_verts = []
        self._roi_line = None
        self._roi_bg = None  # blitting background
        self._roi_labels_layer = None
        self._roi_label_data = None
        self._roi_point_labels = None
        self._roi_mpl_cids = []
        self._roi_g = None
        self._roi_s = None
        self._roi_scatter = None
        self._roi_pixel_yx = None

        self._build_ui()
        self.state.dataset_changed.connect(self._on_dataset_changed)
        self.state.calib_factor_changed.connect(self._on_state_calib_changed)
        self.viewer.layers.events.inserted.connect(
            lambda _: self._refresh_mask_layers()
        )
        self.viewer.layers.events.removed.connect(
            lambda _: self._refresh_mask_layers()
        )

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Toolbar ────────────────────────────────────────────────────
        tb = QHBoxLayout()
        self._plot_btn = QPushButton("Plot")
        self._plot_btn.setEnabled(False)
        tb.addWidget(self._plot_btn)

        self._stale = QLabel("●")
        self._stale.setFixedWidth(14)
        self._stale.setAlignment(Qt.AlignCenter)
        self._stale.setStyleSheet(S.STALE_INACTIVE)
        self._stale.setVisible(False)
        tb.addWidget(self._stale)

        self._roi_btn = QPushButton("⬡ ROI Select")
        self._roi_btn.setCheckable(True)
        self._roi_btn.setToolTip(
            "Toggle interactive lasso ROI selection on the phasor plot"
        )
        self._roi_btn.setEnabled(False)  # enabled after first plot
        self._roi_btn.setVisible(
            False
        )  # TODO: re-enable when ROI Select is ready
        tb.addWidget(self._roi_btn)

        self._roi_done_btn = QPushButton("✓ Done")
        self._roi_done_btn.setVisible(False)
        self._roi_done_btn.setStyleSheet(S.BTN_SUCCESS)
        tb.addWidget(self._roi_done_btn)

        self._roi_btn.toggled.connect(self._on_roi_toggled)
        self._roi_done_btn.clicked.connect(self._on_roi_done)

        tb.addStretch()

        # Monoexponential-lifetime reference overlay — short label here (between
        # Plot and Save), full explanation in the tooltip.
        self._lifetimes_check = QCheckBox("τ marks")
        self._lifetimes_check.setToolTip(
            "Overlay single-exponential lifetime reference marks on the "
            "universal semicircle (…, 1 ns, 2 ns, 3 ns …) for the current "
            "repetition rate, so apparent lifetimes can be read off the plot."
        )
        tb.addWidget(self._lifetimes_check)
        tb.addStretch()
        self._export_combo = QComboBox()
        self._export_combo.addItem("Save plot…", "plot")
        self._export_combo.addItem("Save table…", "table")
        tb.addWidget(self._export_combo)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_export)
        tb.addWidget(self._save_btn)

        root.addLayout(tb)

        # ── Mode — row 1: Per Object / Per Pixel; row 2: per-pixel display ─
        mode_box = QGroupBox("Mode")
        apply_style(mode_box, S.GROUP_PRIMARY)
        mode_outer = QVBoxLayout(mode_box)
        mode_outer.setSpacing(3)  # minimal gap between the two rows
        mode_outer.setContentsMargins(6, 2, 6, 2)

        # Row 1 — object vs pixel
        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)
        self._per_object_radio = QRadioButton("Per Object")
        self._per_pixel_radio = QRadioButton("Per Pixel")
        self._per_object_radio.setChecked(True)
        mode_row.addWidget(self._per_object_radio)
        mode_row.addWidget(self._per_pixel_radio)
        mode_row.addStretch()
        self._per_object_radio.toggled.connect(
            lambda _: self._cancel_roi_if_active()
        )
        self._per_pixel_radio.toggled.connect(
            lambda _: self._cancel_roi_if_active()
        )
        mode_outer.addLayout(mode_row)

        # Row 2 — per-pixel display controls (enabled only in Per Pixel mode)
        px_row = QHBoxLayout()
        px_row.setSpacing(6)
        self._pixel_mode_group = QButtonGroup(self)
        self._pm_scatter = QRadioButton("Scatter")
        self._pm_intensity = QRadioButton("Intensity α")
        self._pm_density = QRadioButton("Density")
        self._pm_scatter.setChecked(True)
        self._pm_scatter.setToolTip(
            "Scatter plot — each pixel as a point (white)"
        )
        self._pm_intensity.setToolTip(
            "Scatter with alpha-channel weighted by photon count\n"
            "(brighter = more photons, color stays the same)"
        )
        self._pm_density.setToolTip(
            "2D histogram density (imshow / hexbin, uses Cmap)"
        )
        for rb, mode_id in (
            (self._pm_scatter, self.PM_SCATTER),
            (self._pm_intensity, self.PM_INTENSITY),
            (self._pm_density, self.PM_DENSITY),
        ):
            self._pixel_mode_group.addButton(rb, mode_id)
            px_row.addWidget(rb)

        self._hexbin_check = QCheckBox("Hex")
        self._hexbin_check.setToolTip(
            "Density bin shape — unchecked: square histogram (imshow); "
            "checked: hexagonal bins (hexbin). Density mode only."
        )
        px_row.addWidget(self._hexbin_check)

        # px_row.addWidget(QLabel("Cmap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(
            ["hot", "inferno", "plasma", "viridis", "magma", "rainbow"]
        )
        self._cmap_combo.setMaximumWidth(85)
        px_row.addWidget(self._cmap_combo)
        px_row.addStretch()
        mode_outer.addLayout(px_row)

        self._pixel_submodes = [
            self._pm_scatter,
            self._pm_intensity,
            self._pm_density,
        ]
        root.addWidget(mode_box)

        # ── Mask ───────────────────────────────────────────────────────
        mask_box = QGroupBox("Mask")
        apply_style(mask_box, S.GROUP_PRIMARY)
        mask_lay = QHBoxLayout(mask_box)
        mask_lay.setSpacing(6)
        mask_lay.setContentsMargins(6, 2, 6, 2)
        mask_lay.addWidget(QLabel("Mask layer:"))
        self._mask_combo = QComboBox()
        self._mask_combo.addItem("None")
        self._mask_combo.setMinimumWidth(100)
        # Give the combo the row's spare width (stretch factor 1) instead of an
        # addStretch() spacer, so it expands from the label to the end and shows
        # long layer names — no fixed width.
        self._mask_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        mask_lay.addWidget(self._mask_combo, 1)

        self._mask_filter_check = QCheckBox("As filter")
        self._mask_filter_check.setToolTip(
            "Treat the mask as binary: plot only the pixels inside it, but as "
            "one population instead of colouring per label.\n"
            "That makes the Per Pixel sub-modes (Intensity α, Density) usable "
            "with a mask."
        )
        mask_lay.addWidget(self._mask_filter_check)
        root.addWidget(mask_box)

        # -- Smoothing (own row, above Calibration) --
        smooth_box = QGroupBox("Smoothing")
        apply_style(smooth_box, S.GROUP_CHECKABLE)
        smooth_box.setCheckable(True)
        smooth_box.setChecked(False)
        smooth_lay = QFormLayout(smooth_box)
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(3, 19)
        self._smooth_spin.setSingleStep(2)  # odd kernel sizes only
        self._smooth_spin.setValue(3)
        kernel_row = QHBoxLayout()
        kernel_row.setContentsMargins(0, 0, 0, 0)
        kernel_row.addWidget(self._smooth_spin)
        kernel_row.addStretch()  # keep the spinbox compact
        smooth_lay.addRow("Kernel:", kernel_row)
        self._smooth_group = smooth_box
        root.addWidget(smooth_box)

        # ── Calibration ────────────────────────────────────────────────
        cal_box = QGroupBox("Calibration")
        apply_style(cal_box, S.GROUP_CHECKABLE)
        cal_box.setCheckable(True)
        cal_box.setChecked(False)
        cal_lay = QHBoxLayout(cal_box)
        cal_lay.setSpacing(4)
        cal_lay.setContentsMargins(6, 4, 6, 4)

        cal_lay.addWidget(QLabel("Factor:"))
        self._calib_display = QLineEdit(self._fmt_complex(self._calib_factor))
        self._calib_display.setReadOnly(True)
        self._calib_display.setStyleSheet(S.DISPLAY)
        self._calib_display.setToolTip(
            "Current calibration factor (complex number). Applied as: phasor × factor"
        )
        self._calib_display.setMaximumWidth(160)
        cal_lay.addWidget(self._calib_display)

        # hidden τ_ref — kept for auto-calibrate; value set via Calculate dialog
        self._tau_ref_spin = QDoubleSpinBox()
        self._tau_ref_spin.setRange(0.001, 1000.0)
        self._tau_ref_spin.setValue(3.834)
        self._tau_ref_spin.setDecimals(3)
        self._tau_ref_spin.setVisible(False)

        calc_btn = QPushButton("Calculate")
        calc_btn.setToolTip(
            "Open dialog: enter τ_ref (ns) + measured G, S → computes factor"
        )
        calc_btn.clicked.connect(self._on_calculate_cal_dialog)
        cal_lay.addWidget(calc_btn)

        custom_btn = QPushButton("Custom")
        custom_btn.setToolTip(
            "Enter a custom calibration factor directly (real + imaginary parts)"
        )
        custom_btn.clicked.connect(self._on_custom_factor_dialog)
        cal_lay.addWidget(custom_btn)

        self._revert_btn = QPushButton("Revert")
        self._revert_btn.clicked.connect(self._revert_calib)
        cal_lay.addWidget(self._revert_btn)
        self._update_revert_tooltip()

        unity_btn = QPushButton("1+0j")
        unity_btn.setToolTip(
            "Set the calibration factor to 1+0j (leaves the phasor unchanged)"
        )
        unity_btn.clicked.connect(self._reset_calib)
        cal_lay.addWidget(unity_btn)

        cal_lay.addStretch()
        self._cal_group = cal_box
        root.addWidget(cal_box)

        # ── Canvas + table splitter ────────────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        cw = QWidget()
        self._plot_area = cw  # hidden until the first plot is drawn
        cw_lay = QVBoxLayout(cw)
        cw_lay.setContentsMargins(0, 0, 0, 0)
        # Small default size (keeps the initial dock narrow); the Expanding
        # size policy + widgetResizable scroll area stretch it to fill the tab.
        self._fig = Figure(figsize=(3.6, 3.6), tight_layout=True)
        self._fig.patch.set_facecolor(MPL.FIG_BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        cw_lay.addWidget(self._canvas)
        splitter.addWidget(cw)
        cw.setVisible(False)  # revealed on the first successful plot

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            [
                "Color",
                "Label",
                "Pixels",
                "Mean G",
                "Mean S",
                "Mean τ (ns)",
                "Total Int.",
            ]
        )
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Fixed
        )
        self._table.setColumnWidth(0, 24)
        self._table.setFixedHeight(130)
        self._table.setVisible(False)
        splitter.addWidget(self._table)

        root.addWidget(splitter, stretch=1)

        # ── Status ─────────────────────────────────────────────────────
        self._status = StatusLabel(
            "Load and reconstruct a PTU file with 'All' output to enable phasor."
        )
        root.addWidget(self._status)

        # Initial empty axes
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(MPL.AXES_BG)
        self._draw_empty()

        # ── Signal wiring ──────────────────────────────────────────────
        self._plot_btn.clicked.connect(self._on_compute_clicked)

        for w in [
            self._per_object_radio,
            self._per_pixel_radio,
            self._pm_scatter,
            self._pm_intensity,
            self._pm_density,
            self._hexbin_check,
            self._mask_combo,
            self._mask_filter_check,
            self._cmap_combo,
            self._lifetimes_check,
            self._smooth_group,
            self._smooth_spin,
            self._cal_group,
        ]:
            if isinstance(w, (QRadioButton, QCheckBox, QGroupBox)):
                w.toggled.connect(self._on_setting_changed)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._on_setting_changed)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self._on_setting_changed)

        self._per_pixel_radio.toggled.connect(self._update_control_states)
        self._mask_combo.currentTextChanged.connect(
            self._update_control_states
        )
        self._pm_density.toggled.connect(self._update_control_states)
        self._hexbin_check.toggled.connect(self._update_control_states)
        self._mask_filter_check.toggled.connect(self._update_control_states)

        self._update_control_states()

    # ------------------------------------------------------------------ #
    # State management                                                     #
    # ------------------------------------------------------------------ #

    def _update_control_states(self):
        """Enable/disable controls based on current mode + mask selection."""
        mask_active = self._mask_combo.currentText() != "None"
        per_pixel = self._per_pixel_radio.isChecked()
        # "As filter" turns a label mask into a binary one: same pixels, but
        # one population — which is what the sub-modes need. Per Pixel only.
        self._mask_filter_check.setEnabled(per_pixel and mask_active)
        color_by_label = mask_active and not self._mask_as_filter()

        # Pixel sub-modes: whenever points are not coloured per label
        sub_modes_ok = per_pixel and not color_by_label
        for rb in self._pixel_submodes:
            rb.setEnabled(sub_modes_ok)

        # Cmap + hex bin: only for density
        density = sub_modes_ok and self._pm_density.isChecked()
        self._cmap_combo.setEnabled(density)
        self._hexbin_check.setEnabled(density)

    def _mask_as_filter(self) -> bool:
        """Mask set to act as a plain binary filter — Per Pixel mode only."""
        return (
            self._per_pixel_radio.isChecked()
            and self._mask_combo.currentText() != "None"
            and self._mask_filter_check.isChecked()
        )

    def _on_setting_changed(self, *_):
        self._update_control_states()
        self._mark_stale()

    def _mark_stale(self):
        if self._plotted_settings is not None:
            if self._get_settings() != self._plotted_settings:
                self._stale.setStyleSheet(S.STALE_STALE)
                self._stale.setToolTip(
                    "Settings or view have changed since the last plot — click Plot to refresh."
                )
                self._stale.setVisible(True)

    def _get_settings(self) -> dict:
        return {
            **self._current_view,
            "per_object": self._per_object_radio.isChecked(),
            "pixel_mode": self._pixel_mode_group.checkedId(),
            "mask": self._mask_combo.currentText(),
            "mask_filter": self._mask_filter_check.isChecked(),
            "cmap": self._cmap_combo.currentText(),
            "smooth": self._smooth_group.isChecked(),
            "smooth_k": self._smooth_spin.value(),
            "calibrate": self._cal_group.isChecked(),
            "calib_re": self._calib_factor.real,
            "calib_im": self._calib_factor.imag,
            "lifetimes": self._lifetimes_check.isChecked(),
            "hexbin": self._hexbin_check.isChecked(),
        }

    # ------------------------------------------------------------------ #
    # Dataset / view signals                                               #
    # ------------------------------------------------------------------ #

    @Slot()
    def _on_dataset_changed(self):
        ds = self.state.dataset
        has_phasor = ds is not None and "phasor_g" in ds and "phasor_s" in ds
        self._plot_btn.setEnabled(has_phasor)
        self._plotted_settings = None
        self._stale.setVisible(False)
        self._final_plot_data = None
        self._export_data = None
        self._plot_area.setVisible(False)  # hide until (re)plotted
        if not has_phasor:
            self._status.info("No phasor data. Reconstruct with 'All' output.")
            self._draw_empty()
        else:
            self._status.info("Phasor data ready. Click 'Plot'.")

    @Slot(dict)
    def on_view_changed(self, settings: dict):
        self._current_view = settings
        self._mark_stale()

    # ------------------------------------------------------------------ #
    # Mask layers                                                          #
    # ------------------------------------------------------------------ #

    def _refresh_mask_layers(self):
        cur = self._mask_combo.currentText()
        self._mask_combo.blockSignals(True)
        self._mask_combo.clear()
        self._mask_combo.addItem("None")
        for layer in self.viewer.layers:
            if type(layer).__name__ == "Labels":
                self._mask_combo.addItem(layer.name)
        idx = self._mask_combo.findText(cur)
        self._mask_combo.setCurrentIndex(max(0, idx))
        self._mask_combo.blockSignals(False)
        self._update_control_states()

    # ------------------------------------------------------------------ #
    # Calibration                                                          #
    # ------------------------------------------------------------------ #

    def _on_calculate_cal_dialog(self):
        """Open dialog: user enters τ_ns + measured G, S → compute factor."""
        dlg = CalibrationDialog(
            default_tau=self._tau_ref_spin.value(), parent=self
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        vals = dlg.get_values()
        try:
            # Persist τ_ref for subsequent Auto calls
            self._tau_ref_spin.setValue(vals["tau_ns"])
            f_rep_hz = self.state.frep_mhz * 1e6
            measured = complex(vals["g"], vals["s"])
            self._calib_factor = _calc_calib_factor(
                vals["tau_ns"], measured, f_rep_hz
            )
            self._calib_display.setText(self._fmt_complex(self._calib_factor))
            self.state.set_calib_factor(self._calib_factor)
            self._mark_stale()
        except Exception as e:
            self._status.error(f"Calibration error: {e}")

    def _on_custom_factor_dialog(self):
        """Open dialog: user types real + imaginary parts directly."""
        dlg = CustomFactorDialog(self._calib_factor, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        r, i = dlg.get_values()
        self._calib_factor = complex(r, i)
        self._calib_display.setText(self._fmt_complex(self._calib_factor))
        self.state.set_calib_factor(self._calib_factor)
        self._mark_stale()

    @Slot(object)
    def _on_state_calib_changed(self, factor):
        """Adopt a factor set elsewhere (e.g. a config loaded in the PTU tab).

        It also becomes the new "Revert" target. The echo of this panel's own
        setters is filtered out (the value is already current).
        """
        factor = complex(factor)
        if factor == self._calib_factor:
            return
        self._loaded_calib_factor = factor
        self._update_revert_tooltip()
        self._calib_factor = factor
        self._calib_display.setText(self._fmt_complex(factor))
        self._mark_stale()

    def _revert_calib(self):
        """Back to the factor loaded with the config (1+0j if none)."""
        self._set_calib_factor(self._loaded_calib_factor)

    def _reset_calib(self):
        """Drop calibration — factor 1+0j."""
        self._set_calib_factor(1.0 + 0j)

    def _set_calib_factor(self, factor: complex):
        self._calib_factor = complex(factor)
        self._calib_display.setText(self._fmt_complex(self._calib_factor))
        self.state.set_calib_factor(self._calib_factor)
        self._mark_stale()

    def _update_revert_tooltip(self):
        self._revert_btn.setToolTip(
            "Restore the calibration factor loaded with the config: "
            f"{self._fmt_complex(self._loaded_calib_factor)}"
        )

    @staticmethod
    def _fmt_complex(c: complex) -> str:
        sign = "+" if c.imag >= 0 else "-"
        return f"{c.real:.4f} {sign} {abs(c.imag):.4f}j"

    # ------------------------------------------------------------------ #
    # Main compute / plot                                                  #
    # ------------------------------------------------------------------ #

    def _on_compute_clicked(self):
        ds = self.state.dataset
        if ds is None or "phasor_g" not in ds:
            return
        self._refresh_mask_layers()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._do_compute(ds)
            self._plot_area.setVisible(True)  # reveal on first/any plot
        except Exception as e:
            traceback.print_exc()
            self._status.error(f"Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _do_compute(self, ds):
        sel, dims_to_sum = _parse_settings(ds, self._current_view)
        sliced = ds.isel(**sel) if sel else ds
        if dims_to_sum:
            from napari_flopa.core.processing.image_utils import (
                aggregate_dataset,
            )

            sliced = aggregate_dataset(sliced, dims_to_sum)

        phasor_g = sliced["phasor_g"].values.squeeze().astype(np.float32)
        phasor_s = sliced["phasor_s"].values.squeeze().astype(np.float32)
        photon_count = (
            sliced["photon_count"].values.squeeze()
            if "photon_count" in sliced
            else None
        )

        ip = ds.attrs.get("instrument_params", {})
        res_ns = ip.get("tcspc_resolution_ns", 1.0)
        lt_2d = (
            sliced["mean_arrival_time"].values.squeeze().astype(np.float32)
            * res_ns
            if "mean_arrival_time" in sliced
            else None
        )

        # ── Smoothing ───────────────────────────────────────────────────
        if self._smooth_group.isChecked() and photon_count is not None:
            k = self._smooth_spin.value()
            if (
                k % 2 == 0
            ):  # tttrkit smooth_phasor shifts ½px with even kernels
                k += 1
            from tttrkit.ptuio.utils import smooth_phasor

            phasor_c_smooth = smooth_phasor(
                phasor_g + 1j * phasor_s,
                photon_count.astype(np.uint32),
                size=k,
            )
            phasor_g = phasor_c_smooth.real
            phasor_s = phasor_c_smooth.imag

        # ── Calibration ─────────────────────────────────────────────────
        phasor_c = phasor_g + 1j * phasor_s
        if self._cal_group.isChecked() and self._calib_factor != (1.0 + 0j):
            phasor_c = phasor_c * self._calib_factor
        phasor_g = phasor_c.real.astype(np.float32)
        phasor_s = phasor_c.imag.astype(np.float32)

        shape_2d = phasor_g.shape

        # ── Mask ────────────────────────────────────────────────────────
        mask_name = self._mask_combo.currentText()
        mask_active = mask_name != "None"
        label_2d = None
        mask_layer = None
        if mask_active and mask_name in self.viewer.layers:
            mask_layer = self.viewer.layers[mask_name]
            ldata = np.asarray(mask_layer.data)
            if ldata.ndim > 2:
                ldata = ldata.reshape(-1, ldata.shape[-2], ldata.shape[-1])[0]
            if ldata.shape == shape_2d:
                label_2d = ldata.astype(np.int32)
            else:
                self._status.warn(
                    f"Mask shape {ldata.shape} ≠ phasor shape {shape_2d}. "
                    "Mask ignored."
                )
                mask_active = False

        per_object = self._per_object_radio.isChecked()
        # Binary-filter mode (Per Pixel only): the mask still restricts which
        # pixels are plotted — that happens below via `valid` — but they are
        # drawn as one population instead of coloured per label.
        # `mask_active` (not the combo) so a shape-mismatched mask, which was
        # just discarded above, cannot leave filter mode switched on.
        mask_as_filter = mask_active and self._mask_as_filter()
        color_by_label = mask_active and not mask_as_filter

        # ── Build final g/s arrays ──────────────────────────────────────
        (
            g_out,
            s_out,
            labels_out,
            photons_out,
            areas_out,
            colors_out,
            lt_out,
        ) = ([], [], [], [], [], [], [])

        if per_object:
            from tttrkit.ptuio.utils import average_phasor

            phasor_c = phasor_g + 1j * phasor_s
            pc_arr = (
                photon_count.astype(np.float64)
                if photon_count is not None
                else np.ones(phasor_c.shape)
            )

            if not mask_active:
                # whole image → single centroid
                avg = average_phasor(phasor_c, pc_arr)
                valid = (
                    np.isfinite(phasor_g)
                    & np.isfinite(phasor_s)
                    & (pc_arr > 0)
                )
                g_out.append(float(avg.real))
                s_out.append(float(avg.imag))
                labels_out.append(np.nan)
                photons_out.append(float(pc_arr[valid].sum()))
                areas_out.append(int(valid.sum()))
                colors_out.append((1.0, 1.0, 1.0, 0.9))  # white
                lt_v = lt_2d[valid] if lt_2d is not None else None
                lt_out.append(
                    float(np.nanmean(lt_v))
                    if lt_v is not None
                    else float("nan")
                )
            else:
                for lbl in np.unique(label_2d):
                    if lbl == 0:
                        continue
                    m = label_2d == lbl
                    avg = average_phasor(phasor_c, pc_arr, mask=m)
                    if not np.isfinite(avg):
                        continue
                    valid = (
                        m
                        & np.isfinite(phasor_g)
                        & np.isfinite(phasor_s)
                        & (pc_arr > 0)
                    )
                    g_out.append(float(avg.real))
                    s_out.append(float(avg.imag))
                    labels_out.append(int(lbl))
                    photons_out.append(float(pc_arr[valid].sum()))
                    areas_out.append(int(m.sum()))
                    lt_v = lt_2d[valid] if lt_2d is not None else None
                    lt_out.append(
                        float(np.nanmean(lt_v))
                        if lt_v is not None
                        else float("nan")
                    )
                    rgba = _get_napari_color(mask_layer, int(lbl))
                    colors_out.append(rgba)

        else:  # per pixel
            valid = np.isfinite(phasor_g) & np.isfinite(phasor_s)
            if photon_count is not None:
                valid &= photon_count > 0
            if mask_active and label_2d is not None:
                valid &= label_2d > 0

            yx_valid = np.argwhere(
                valid
            )  # (N, 2) — row, col of every valid pixel

            g_pix = phasor_g[valid]
            s_pix = phasor_s[valid]
            pc_pix = (
                photon_count[valid].astype(np.float32)
                if photon_count is not None
                else None
            )
            lt_pix = lt_2d[valid] if lt_2d is not None else None
            lbl_pix = (
                label_2d[valid]
                if (mask_active and label_2d is not None)
                else None
            )

            # Save full arrays for table stats (before subsampling)
            g_full = g_pix
            s_full = s_pix
            pc_full = pc_pix
            lt_full = lt_pix
            lbl_full = lbl_pix

            # sub-sample if too large (for plotting only)
            if len(g_pix) > _MAX_PX:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(g_pix), _MAX_PX, replace=False)
                g_pix = g_pix[idx]
                s_pix = s_pix[idx]
                pc_pix = pc_pix[idx] if pc_pix is not None else None
                lt_pix = lt_pix[idx] if lt_pix is not None else None
                lbl_pix = lbl_pix[idx] if lbl_pix is not None else None
                yx_plot = yx_valid[idx]
            else:
                yx_plot = yx_valid

            self._roi_pixel_yx = (yx_plot[:, 0], yx_plot[:, 1])

            g_out = g_pix
            s_out = s_pix
            photons_out = pc_pix if pc_pix is not None else np.ones(len(g_pix))
            labels_out = (
                lbl_pix if lbl_pix is not None else np.full(len(g_pix), np.nan)
            )
            areas_out = np.full(len(g_pix), np.nan)
            lt_out = (
                lt_pix if lt_pix is not None else np.full(len(g_pix), np.nan)
            )

            if color_by_label and lbl_pix is not None:
                # build RGBA per point from napari label colors — one lookup
                # per plotted point, so skip it entirely in filter mode
                colors_out = np.array(
                    [
                        _get_napari_color(mask_layer, int(lbl))
                        for lbl in lbl_pix
                    ]
                )  # shape (N, 4)
            else:
                colors_out = "white"

        # ── Store the drawn points (ROI selection indexes into these) ───
        dataset_name = ds.attrs.get("source_filename", "N/A")
        self._final_plot_data = {
            "g_coords": np.array(g_out, dtype=np.float32).ravel(),
            "s_coords": np.array(s_out, dtype=np.float32).ravel(),
            "photon_counts": np.array(photons_out, dtype=np.float32).ravel(),
            "labels": np.array(labels_out).ravel(),
            "areas": np.array(areas_out).ravel(),
            "dataset_name": dataset_name,
        }

        # ── Store for CSV export — every valid point, including the ones
        #    the plot dropped when subsampling ─────────────────────────────
        if per_object:
            self._export_data = self._final_plot_data
        else:
            n_full = len(g_full)
            self._export_data = {
                "g_coords": np.asarray(g_full, dtype=np.float32).ravel(),
                "s_coords": np.asarray(s_full, dtype=np.float32).ravel(),
                "photon_counts": (
                    np.asarray(pc_full, dtype=np.float32).ravel()
                    if pc_full is not None
                    else np.ones(n_full, dtype=np.float32)
                ),
                "labels": (
                    np.asarray(lbl_full).ravel()
                    if lbl_full is not None
                    else np.full(n_full, np.nan)
                ),
                "areas": np.full(n_full, np.nan),
                "dataset_name": dataset_name,
            }

        # ── Draw ────────────────────────────────────────────────────────
        if per_object:
            self._roi_pixel_yx = None
        sync_hz = ip.get("sync_rate_hz", None) or (self.state.frep_mhz * 1e6)
        self._draw_phasor(
            np.array(g_out).ravel(),
            np.array(s_out).ravel(),
            colors_out,
            sync_hz=float(sync_hz),
            per_object=per_object,
            color_by_label=color_by_label,
            photon_counts=(
                np.array(photons_out).ravel() if not per_object else None
            ),
            labels=np.array(labels_out).ravel(),
            mask_layer=mask_layer,
            lt_arr=np.array(lt_out).ravel(),
        )

        # ── Table ────────────────────────────────────────────────────────
        if per_object:
            self._fill_table_per_object(
                np.array(g_out),
                np.array(s_out),
                np.array(lt_out),
                np.array(photons_out),
                np.array(labels_out),
                colors_out,
            )
        else:
            self._fill_table_summary(
                g_full,
                s_full,
                lt_full,
                pc_full,
                # Filter mode → no labels, so the table summarises the masked
                # pixels as the single population the plot shows.
                None if mask_as_filter else lbl_full,
                mask_layer,
            )
        self._table.setVisible(True)

        self._plotted_settings = self._get_settings()
        self._stale.setStyleSheet(S.STALE_FRESH)
        self._stale.setToolTip("Plot is up to date.")
        self._stale.setVisible(True)
        self._roi_btn.setEnabled(True)
        n_plot = len(np.array(g_out).ravel())
        if per_object:
            self._status.info(f"Plotted {n_plot:,} objects.")
        else:
            n_full = len(g_full)
            scope = " inside the mask" if mask_as_filter else ""
            if n_plot < n_full:
                self._status.warn(
                    f"Plot subsampled: {n_plot:,} of {n_full:,} valid pixels"
                    f"{scope} drawn. 'Save table' still exports all of them."
                )
            else:
                self._status.info(f"Plotted {n_plot:,} pixels{scope}.")

    # ------------------------------------------------------------------ #
    # Drawing                                                              #
    # ------------------------------------------------------------------ #

    def _draw_phasor(
        self,
        g,
        s,
        colors,
        *,
        sync_hz,
        per_object,
        color_by_label,
        photon_counts,
        labels,
        mask_layer,
        lt_arr,
    ):
        """Draw the phasor plot.

        *color_by_label* means "a mask is active and should tint the points";
        it is False when the mask acts as a plain binary filter, which routes
        per-pixel drawing to the Scatter / Intensity / Density sub-modes.
        """
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._ax = ax
        ax.set_facecolor(MPL.AXES_BG)
        fig.patch.set_facecolor(MPL.FIG_BG)

        # Universal semicircle
        theta = np.linspace(0, np.pi, 300)
        ax.plot(
            0.5 + 0.5 * np.cos(theta),
            0.5 * np.sin(theta),
            color="white",
            linewidth=0.8,
            alpha=0.5,
            zorder=5,
        )

        if self._lifetimes_check.isChecked():
            _draw_lifetime_ticks(ax, sync_hz)

        if len(g) == 0:
            self._finalise_axes(ax)
            self._canvas.draw_idle()
            return

        if per_object:
            # Only centroids — no background scatter
            marker_size = 60 if len(g) == 1 else 50
            ax.scatter(
                g,
                s,
                c=colors,
                s=marker_size,
                edgecolors="white",
                linewidths=0.6,
                alpha=0.9,
                zorder=10,
            )
            # Legend (≤12 labels)
            if color_by_label and labels is not None and len(g) <= 12:
                import matplotlib.patches as mpatches

                handles = []
                for i, lbl in enumerate(labels):
                    fc = (
                        colors[i]
                        if isinstance(colors, list)
                        else (1, 1, 1, 0.9)
                    )
                    lbl_str = f"#{int(lbl)}" if np.isfinite(lbl) else "All"
                    handles.append(mpatches.Patch(color=fc, label=lbl_str))
                ax.legend(
                    handles=handles,
                    fontsize=7,
                    loc="upper right",
                    facecolor=MPL.AXES_BG,
                    edgecolor=MPL.SPINE,
                    labelcolor=MPL.TICK,
                    framealpha=0.8,
                )

        elif color_by_label:
            # Per pixel + mask → color by label
            ax.scatter(g, s, c=colors, s=2, alpha=0.4, linewidths=0, zorder=1)
            # Legend (≤12 unique labels)
            unique_lbls = (
                np.unique(labels[np.isfinite(labels)].astype(int))
                if labels is not None
                else []
            )
            if (
                len(unique_lbls) <= 12
                and len(unique_lbls) > 0
                and mask_layer is not None
            ):
                import matplotlib.patches as mpatches

                handles = []
                for lbl in unique_lbls:
                    rgba = _get_napari_color(mask_layer, int(lbl))
                    handles.append(mpatches.Patch(color=rgba, label=f"#{lbl}"))
                ax.legend(
                    handles=handles,
                    fontsize=7,
                    loc="upper right",
                    facecolor=MPL.AXES_BG,
                    edgecolor=MPL.SPINE,
                    labelcolor=MPL.TICK,
                    framealpha=0.8,
                )

        else:
            # Per pixel, no mask — use sub-mode
            pixel_mode = self._pixel_mode_group.checkedId()

            if pixel_mode == self.PM_SCATTER:
                ax.scatter(
                    g, s, c="white", s=2, alpha=0.5, linewidths=0, zorder=1
                )

            elif pixel_mode == self.PM_INTENSITY:
                # Alpha modulated by photon count — color stays white (or label color)
                if photon_counts is not None and photon_counts.max() > 0:
                    alpha_vals = np.clip(
                        photon_counts / photon_counts.max(), 0.02, 1.0
                    )
                else:
                    alpha_vals = np.full(len(g), 0.5)
                # Build RGBA array (white with per-point alpha)
                rgba = np.ones((len(g), 4), dtype=np.float32)
                rgba[:, 3] = alpha_vals
                ax.scatter(g, s, c=rgba, s=2, linewidths=0, zorder=1)

            elif pixel_mode == self.PM_DENSITY:
                cmap_name = self._cmap_combo.currentText()
                if self._hexbin_check.isChecked():
                    hb = ax.hexbin(
                        g,
                        s,
                        gridsize=60,
                        cmap=cmap_name,
                        mincnt=1,
                        extent=[-0.1, 1.1, -0.1, 0.85],
                        zorder=1,
                    )
                    cb = fig.colorbar(
                        hb,
                        ax=ax,
                        orientation="horizontal",
                        fraction=0.046,
                        pad=0.12,
                    )
                else:
                    H, xedges, yedges = np.histogram2d(
                        g, s, bins=150, range=[[-0.1, 1.1], [-0.1, 0.85]]
                    )
                    alpha_img = (H > 0).astype(float)
                    im = ax.imshow(
                        H.T,
                        origin="lower",
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        cmap=cmap_name,
                        aspect="auto",
                        alpha=alpha_img.T,
                        zorder=1,
                    )
                    cb = fig.colorbar(
                        im,
                        ax=ax,
                        orientation="horizontal",
                        fraction=0.046,
                        pad=0.12,
                    )
                cb.ax.xaxis.set_tick_params(colors=MPL.TICK, labelsize=7)
                cb.ax.set_facecolor(MPL.AXES_BG)
                cb.outline.set_edgecolor(MPL.SPINE)
                cb.set_label("Count", color=MPL.TICK, fontsize=8)

        self._finalise_axes(ax)

        # Store G/S for ROI selection (_roi_pixel_yx set by _do_compute)
        self._roi_g = np.asarray(g, dtype=np.float32).ravel()
        self._roi_s = np.asarray(s, dtype=np.float32).ravel()
        self._roi_scatter = None

        self._canvas.draw_idle()

    def _finalise_axes(self, ax):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.80)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("G", color=MPL.TICK)
        ax.set_ylabel("S", color=MPL.TICK)
        ax.tick_params(colors=MPL.TICK, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(MPL.SPINE)

    def _draw_empty(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        self._ax = ax
        ax.set_facecolor(MPL.AXES_BG)
        self._fig.patch.set_facecolor(MPL.FIG_BG)
        theta = np.linspace(0, np.pi, 300)
        ax.plot(
            0.5 + 0.5 * np.cos(theta),
            0.5 * np.sin(theta),
            color="#555555",
            linewidth=0.8,
        )
        ax.text(
            0.5,
            0.25,
            "No data",
            ha="center",
            va="center",
            color="#555555",
            fontsize=11,
        )
        self._finalise_axes(ax)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Table                                                                #
    # ------------------------------------------------------------------ #

    def _fill_table_per_object(self, g, s, lt, photons, labels, colors):
        self._table.setRowCount(len(g))
        for r in range(len(g)):
            rgba = colors[r] if isinstance(colors, list) else (1, 1, 1)
            self._set_table_row(
                r,
                rgba,
                str(int(labels[r])) if np.isfinite(labels[r]) else "all",
                int(photons[r]) if np.isfinite(photons[r]) else 0,
                g[r],
                s[r],
                (
                    lt[r]
                    if lt is not None and np.isfinite(lt[r])
                    else float("nan")
                ),
                float(photons[r]),
            )
        self._table.resizeColumnsToContents()
        self._table.setColumnWidth(0, 24)

    def _fill_table_summary(self, g, s, lt, photons, labels, mask_layer):
        """For per-pixel mode: one row per unique label (or one row for whole image).
        Receives full (non-subsampled) arrays for accurate statistics."""
        if g is None or len(g) == 0:
            self._table.setRowCount(0)
            return
        unique = (
            np.unique(labels[np.isfinite(labels)].astype(int))
            if labels is not None and np.any(np.isfinite(labels))
            else []
        )
        if len(unique) == 0:
            # Whole image summary — intensity-weighted means to match per-object mode
            self._table.setRowCount(1)
            w = photons.astype(np.float64) if photons is not None else None
            wsum = float(np.nansum(w)) if w is not None else 0.0
            if wsum > 0:
                g_val = float(np.nansum(g.astype(np.float64) * w) / wsum)
                s_val = float(np.nansum(s.astype(np.float64) * w) / wsum)
                lt_val = (
                    float(np.nansum(lt.astype(np.float64) * w) / wsum)
                    if lt is not None
                    else float("nan")
                )
            else:
                g_val = float(np.nanmean(g))
                s_val = float(np.nanmean(s))
                lt_val = (
                    float(np.nanmean(lt)) if lt is not None else float("nan")
                )
            self._set_table_row(
                0,
                (1.0, 1.0, 1.0),
                "All",
                len(g),
                g_val,
                s_val,
                lt_val,
                wsum if wsum > 0 else float(len(g)),
            )
        else:
            self._table.setRowCount(len(unique))
            for r, lbl in enumerate(unique):
                m = labels == lbl
                rgba = (
                    _get_napari_color(mask_layer, int(lbl))
                    if mask_layer
                    else (0.5, 0.5, 0.5)
                )
                self._set_table_row(
                    r,
                    rgba,
                    str(lbl),
                    int(m.sum()),
                    float(np.nanmean(g[m])),
                    float(np.nanmean(s[m])),
                    (
                        float(np.nanmean(lt[m]))
                        if lt is not None
                        else float("nan")
                    ),
                    (
                        float(np.nansum(photons[m]))
                        if photons is not None
                        else float(m.sum())
                    ),
                )
        self._table.resizeColumnsToContents()
        self._table.setColumnWidth(0, 24)

    def _set_table_row(
        self, r, rgba, label_str, pixels, mean_g, mean_s, mean_lt, total_int
    ):
        swatch = QTableWidgetItem()
        swatch.setBackground(
            QBrush(
                QColor(
                    int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                )
            )
        )
        swatch.setFlags(Qt.ItemIsEnabled)
        self._table.setItem(r, 0, swatch)
        for c, val in enumerate(
            [
                label_str,
                str(pixels),
                f"{mean_g:.4f}",
                f"{mean_s:.4f}",
                f"{mean_lt:.3f}",
                f"{total_int:.1f}",
            ]
        ):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(r, 1 + c, item)

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def _on_export(self):
        fmt = self._export_combo.currentData()
        if fmt == "plot":
            self._save_plot()
        else:
            self._save_table()

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Phasor Plot",
            "phasor.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        try:
            self._fig.savefig(
                path, dpi=150, facecolor=MPL.FIG_BG, bbox_inches="tight"
            )
            self._status.info(f"Plot saved: {Path(path).name}")
        except Exception as e:
            self._status.error(f"Save error: {e}")

    def _save_table(self):
        # Exports every valid point, not just the subsample that was drawn.
        if not self._export_data:
            self._status.info("No data — click 'Plot' first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Table", "phasor_table.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            import pandas as pd

            d = self._export_data
            pd.DataFrame(
                {
                    "dataset_name": d["dataset_name"],
                    "label_id": d["labels"],
                    "g": d["g_coords"],
                    "s": d["s_coords"],
                    "photon_count_sum": d["photon_counts"],
                    "area_pixels": d["areas"],
                }
            ).to_csv(path, index=False)
            self._status.info(
                f"Table saved: {Path(path).name} ({len(d['g_coords']):,} rows)"
            )
        except Exception as e:
            self._status.error(f"Save error: {e}")

    # ── ROI Selection ──────────────────────────────────────────────────────

    def hideEvent(self, event):
        """Cancel ROI selection when this panel is hidden (e.g. tab switch)."""
        self._cancel_roi_if_active()
        super().hideEvent(event)

    def _cancel_roi_if_active(self):
        if self._roi_active:
            self._on_roi_done()

    def _on_roi_toggled(self, checked: bool):
        """Toggle interactive lasso ROI mode on/off."""
        self._roi_active = checked
        self._roi_btn.setText(
            "⬡ ROI Select" if not checked else "● ROI Active"
        )
        self._roi_done_btn.setVisible(checked)
        if checked:
            # Connect matplotlib mouse events
            canvas = self._canvas
            self._roi_mpl_cids = [
                canvas.mpl_connect("button_press_event", self._roi_on_press),
                canvas.mpl_connect("motion_notify_event", self._roi_on_motion),
                canvas.mpl_connect(
                    "button_release_event", self._roi_on_release
                ),
            ]
        else:
            self._roi_disconnect()

    def _roi_disconnect(self):
        for cid in self._roi_mpl_cids:
            self._canvas.mpl_disconnect(cid)
        self._roi_mpl_cids = []
        self._roi_bg = None
        if self._roi_line is not None:
            with contextlib.suppress(Exception):
                self._roi_line.remove()
            self._roi_line = None
        self._roi_lasso_verts = []
        self._canvas.draw_idle()

    def _on_roi_done(self):
        """Finish ROI session: disconnect events, reset for fresh start."""
        self._roi_btn.setChecked(False)
        self._roi_btn.setText("⬡ ROI Select")
        self._roi_done_btn.setVisible(False)
        self._roi_disconnect()
        self._roi_active = False
        # Reset state so next toggle starts fresh
        self._roi_label_id = 0
        self._roi_point_labels = None
        self._roi_labels_layer = None
        self._roi_label_data = None

    def _roi_on_press(self, event):
        if event.inaxes != self._ax or event.button != 1:
            return
        self._roi_lasso_verts = [(event.xdata, event.ydata)]
        if self._roi_line is not None:
            with contextlib.suppress(Exception):
                self._roi_line.remove()
        (self._roi_line,) = self._ax.plot(
            [event.xdata],
            [event.ydata],
            color="yellow",
            linewidth=1.0,
            alpha=0.8,
            zorder=20,
            animated=True,
        )
        # Capture static background for blitting (excludes animated artists)
        self._canvas.draw()
        self._roi_bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _roi_on_motion(self, event):
        if not self._roi_lasso_verts or event.inaxes != self._ax:
            return
        self._roi_lasso_verts.append((event.xdata, event.ydata))
        xs = [v[0] for v in self._roi_lasso_verts]
        ys = [v[1] for v in self._roi_lasso_verts]
        self._roi_line.set_data(xs, ys)
        # Blit: restore background, draw only the lasso line, flush
        self._canvas.restore_region(self._roi_bg)
        self._ax.draw_artist(self._roi_line)
        self._canvas.blit(self._ax.bbox)

    def _roi_on_release(self, event):
        if event.button != 1:
            return
        if len(self._roi_lasso_verts) < 3:
            self._roi_lasso_verts = []
            self._roi_bg = None
            return
        # Close the polygon
        verts = self._roi_lasso_verts + [self._roi_lasso_verts[0]]
        self._roi_lasso_verts = []
        self._roi_bg = None
        # Switch line back to non-animated so it persists in the normal render
        self._roi_line.set_animated(False)
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        self._roi_line.set_data(xs, ys)

        # Test which plotted points are inside the polygon
        if (
            not hasattr(self, "_roi_g")
            or self._roi_g is None
            or len(self._roi_g) == 0
        ):
            return

        poly = MplPath(verts)
        pts = np.column_stack([self._roi_g, self._roi_s])
        inside = poly.contains_points(pts)

        if not inside.any():
            return

        self._roi_label_id += 1
        lbl = self._roi_label_id

        # Initialise point labels array
        if self._roi_point_labels is None:
            self._roi_point_labels = np.zeros(len(self._roi_g), dtype=np.int32)
        # Overwrite any previous assignment for points inside
        self._roi_point_labels[inside] = lbl

        # Update napari Labels layer first (so colors are available for scatter)
        self._roi_update_labels_layer(inside, lbl)

        # Update scatter colors on plot
        self._roi_update_scatter_colors()

    def _roi_update_scatter_colors(self):
        """Recolor all scatter points: assigned → napari label color, unassigned → gray."""
        if self._roi_point_labels is None or self._roi_labels_layer is None:
            return
        ax = self._ax
        # Remove existing ROI scatter if present
        if hasattr(self, "_roi_scatter") and self._roi_scatter is not None:
            with contextlib.suppress(Exception):
                self._roi_scatter.remove()
        n = len(self._roi_g)
        colors = np.zeros((n, 4), dtype=np.float32)
        colors[:, :3] = 0.35  # gray for unassigned
        colors[:, 3] = 0.5
        for lbl_id in range(1, self._roi_label_id + 1):
            mask = self._roi_point_labels == lbl_id
            if mask.any():
                rgba = _get_napari_color(self._roi_labels_layer, lbl_id)
                colors[mask] = rgba
                colors[mask, 3] = 0.85
        # Determine marker size based on mode
        per_obj = self._per_object_radio.isChecked()
        ms = 50 if per_obj else 3
        self._roi_scatter = ax.scatter(
            self._roi_g,
            self._roi_s,
            c=colors,
            s=ms,
            edgecolors="none" if not per_obj else "white",
            linewidths=0 if not per_obj else 0.5,
            zorder=15,
        )
        self._canvas.draw_idle()

    def _roi_update_labels_layer(self, inside_mask: np.ndarray, lbl: int):
        """
        Paint pixels for the selected phasor points into the Labels layer.

        For per-object mode: inside_mask selects phasor centroids → look up their
        object IDs in self._final_plot_data['labels'] → paint those objects' pixels.
        For per-pixel mode: inside_mask selects individual pixels stored in
        self._roi_pixel_yx → paint those pixels directly.
        """
        ds = self.state.dataset
        if ds is None:
            return

        # Get image shape from dataset
        shape_2d = None
        for var in ("photon_count", "phasor_g", "mean_arrival_time"):
            if var in ds:
                arr = ds[var]
                # squeeze out non-spatial dims to get (lines, pixels)
                vals = arr.values
                # find last 2 dims
                if vals.ndim >= 2:
                    shape_2d = vals.shape[-2:]
                break
        if shape_2d is None:
            return

        # Create or reuse labels layer
        if (
            self._roi_labels_layer is None
            or self._roi_labels_layer not in self.viewer.layers
        ):
            self._roi_label_data = np.zeros(shape_2d, dtype=np.int32)
            try:
                self._roi_labels_layer = self.viewer.add_labels(
                    self._roi_label_data.copy(), name="Phasor ROI Labels"
                )
            except Exception:
                return

        per_obj = self._per_object_radio.isChecked()
        fd = self._final_plot_data

        if per_obj and fd is not None:
            # labels array holds object IDs (from the mask layer used for plotting)
            obj_labels = fd.get("labels")
            if obj_labels is None:
                return
            obj_labels_arr = np.asarray(obj_labels).ravel()
            selected_obj_ids = obj_labels_arr[inside_mask]

            # We need the label_2d map — get it from the current mask layer
            mask_name = self._mask_combo.currentText()
            if mask_name == "None" or mask_name not in self.viewer.layers:
                # No mask → whole image, paint everything with lbl
                self._roi_label_data[:] = lbl
            else:
                mask_layer = self.viewer.layers[mask_name]
                ldata = np.asarray(mask_layer.data)
                if ldata.ndim > 2:
                    ldata = ldata.reshape(
                        -1, ldata.shape[-2], ldata.shape[-1]
                    )[0]
                if ldata.shape != shape_2d:
                    return
                label_2d = ldata.astype(np.int32)
                for oid in selected_obj_ids:
                    if np.isfinite(oid):
                        self._roi_label_data[label_2d == int(oid)] = lbl
        else:
            # Per-pixel: use stored pixel indices
            if (
                hasattr(self, "_roi_pixel_yx")
                and self._roi_pixel_yx is not None
            ):
                ys, xs = self._roi_pixel_yx
                in_ys = ys[inside_mask]
                in_xs = xs[inside_mask]
                valid = (
                    (in_ys >= 0)
                    & (in_ys < shape_2d[0])
                    & (in_xs >= 0)
                    & (in_xs < shape_2d[1])
                )
                self._roi_label_data[in_ys[valid], in_xs[valid]] = lbl

        self._roi_labels_layer.data = self._roi_label_data.copy()


# ── CalibrationDialog ─────────────────────────────────────────────────────


class CalibrationDialog(QDialog):
    """Enter reference lifetime + measured G, S → computes calibration factor."""

    def __init__(self, default_tau: float = 3.834, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculate Calibration Factor")
        lay = QVBoxLayout(self)
        form = QFormLayout()

        self._tau = QDoubleSpinBox()
        self._tau.setRange(0.001, 1000.0)
        self._tau.setDecimals(4)
        self._tau.setValue(default_tau)
        form.addRow("Reference lifetime τ (ns):", self._tau)

        self._g = QDoubleSpinBox()
        self._g.setRange(-2.0, 2.0)
        self._g.setDecimals(5)
        self._g.setSingleStep(0.001)
        form.addRow("Measured phasor G:", self._g)

        self._s = QDoubleSpinBox()
        self._s.setRange(-2.0, 2.0)
        self._s.setDecimals(5)
        self._s.setSingleStep(0.001)
        form.addRow("Measured phasor S:", self._s)

        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_values(self) -> dict:
        return {
            "tau_ns": self._tau.value(),
            "g": self._g.value(),
            "s": self._s.value(),
        }


# ── CustomFactorDialog ────────────────────────────────────────────────────


class CustomFactorDialog(QDialog):
    """Enter a calibration factor directly as real + imaginary parts."""

    def __init__(self, current: complex = 1 + 0j, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Custom Calibration Factor")
        lay = QVBoxLayout(self)
        form = QFormLayout()

        self._real = QDoubleSpinBox()
        self._real.setRange(-100.0, 100.0)
        self._real.setDecimals(6)
        self._real.setSingleStep(0.001)
        self._real.setValue(current.real)
        form.addRow("Real part:", self._real)

        self._imag = QDoubleSpinBox()
        self._imag.setRange(-100.0, 100.0)
        self._imag.setDecimals(6)
        self._imag.setSingleStep(0.001)
        self._imag.setValue(current.imag)
        form.addRow("Imaginary part (j):", self._imag)

        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_values(self) -> tuple:
        return (self._real.value(), self._imag.value())


# ── Module-level helpers ──────────────────────────────────────────────────


def _parse_settings(ds, settings: dict):
    sel, dims_to_sum = {}, []
    for dim in ["frame", "sequence", "channel"]:
        sum_key = f"sum_{dim}s"
        if dim in ds.sizes and settings.get(sum_key, False):
            dims_to_sum.append(dim)
        elif dim in ds.sizes:
            sel[dim] = settings.get(dim, 0)
    return sel, dims_to_sum


def _calc_calib_factor(
    tau_ns: float, measured: complex, f_rep_hz: float
) -> complex:
    tau_s = tau_ns * 1e-9
    omega = 2 * np.pi * f_rep_hz
    g_th = 1.0 / (1.0 + (omega * tau_s) ** 2)
    s_th = (omega * tau_s) / (1.0 + (omega * tau_s) ** 2)
    theory = complex(g_th, s_th)
    if abs(measured) < 1e-12:
        raise ValueError(
            "Measured phasor is zero — cannot compute calibration factor."
        )
    return theory / measured


def _get_napari_color(layer, label_id: int) -> tuple:
    """Return (R, G, B, A) float 0-1 for label_id from a napari Labels layer."""
    try:
        rgba = layer.get_color(label_id)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
    except Exception:
        return (0.5, 0.5, 0.5, 0.8)


def _draw_lifetime_ticks(ax, sync_hz: float):
    omega = 2 * np.pi * sync_hz
    tau_max = max(1, int(np.ceil(0.5e9 / sync_hz)))
    center = np.array([0.5, 0.0])
    tick_len = 0.018
    for tau_ns in range(1, tau_max + 1):
        tau_s = tau_ns * 1e-9
        g_t = 1.0 / (1.0 + (omega * tau_s) ** 2)
        s_t = (omega * tau_s) / (1.0 + (omega * tau_s) ** 2)
        p = np.array([g_t, s_t])
        v = p - center
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        v_u = v / nrm
        p1, p2 = p - tick_len / 2 * v_u, p + tick_len / 2 * v_u
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], "w-", lw=0.8, alpha=0.6, zorder=4
        )
        lp = p + tick_len * 1.7 * v_u
        ax.text(
            lp[0],
            lp[1],
            f"{tau_ns}ns",
            fontsize=6,
            ha="center",
            va="center",
            color="white",
            alpha=0.7,
            zorder=4,
        )
