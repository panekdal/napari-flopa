import traceback
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import colormaps
from qtpy.QtCore import Qt, QTimer, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_flopa.core.processing.image_utils import (
    INTENSITY_COLORMAPS,
    LIFETIME_COLORMAPS,
    aggregate_dataset,
    colormap_to_lut,
    flim_export_info,
    flim_rgb,
    smooth_count,
    smooth_weighted,
)
from napari_flopa.ui.state import FlopaState
from napari_flopa.ui.style import S, apply_style
from napari_flopa.ui.widgets.histogram_slider import HistogramSlider

_CANVAS_H = 55  # histogram canvas height in compact layout
_HIST_MAX_W = 300  # histogram slider max width


class FlimViewPanel(QWidget):
    """
    Bottom dock panel for interactive FLIM image display.

    Provides:
      • Frame / sequence / detector spinbox selectors with per-dim "Agg"
        checkboxes that sum over that dimension.
      • Dual HistogramSlider for Intensity and Lifetime — cyan handles set
        display contrast, red handles define the mask threshold range.
      • Smoothing controls (kernel size) for both intensity and lifetime.
      • Colormap selectors; FLIM RGB composite mode when both layers are ON.
      • "→ Generate Int./Lt. Mask" buttons create uniquely named Labels
        layers from the current red-slider threshold range.
      • Export buttons for Intensity (uint32 photon-count TIFF), Lifetime
        (float32 ns/ch TIFF), and FLIM RGB composite (PNG/TIFF + a .txt sidecar
        recording the colormap and contrast ranges).

    The widget is hidden until update_data() is called with a valid dataset.
    Emits view_changed(dict) whenever the selection or aggregation changes so
    that PhasorPanel and DecayPanel can mark themselves stale.
    """

    view_changed = Signal(dict)

    def __init__(self, state: FlopaState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer
        self.dataset = None
        self._current_intensity = None
        self._current_lifetime = None
        self._lut = None  # 256×3 uint8 lifetime colormap LUT
        # (sel_tuple, frozenset(dims_to_sum)) → (raw_int, raw_lt). Only used
        # when _use_sum_cache is True (off by default; not exposed in the GUI).
        self._sum_cache = {}
        self._use_sum_cache = False  # set True to memoise isel + aggregation
        # self._smooth_cache = {}  # (type, id, shape, k) → smoothed array (caching disabled)
        self._layers_created = False
        self.selectors = {}

        # Debounce timer kept as instance attr so it is not GC-collected
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)

        self._build_ui()
        self.setVisible(False)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 0, 2, 0)

        self.container = QGroupBox("FLIM VIEW")
        apply_style(self.container, S.GROUP_DOCK)
        self.view_layout = QVBoxLayout(self.container)
        self.view_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)
        main_layout.addStretch()

    # ------------------------------------------------------------------ #

    @Slot(object)
    def update_data(self, dataset: xr.Dataset):
        """
        Called after reconstruction completes.  Stores the dataset, clears
        caches, rebuilds all controls via _create_controls(), and makes the
        widget visible.  Hides the widget if neither photon_count nor
        mean_arrival_time is present.
        """
        self.dataset = dataset
        self._sum_cache.clear()  # drop stale entries from the previous dataset
        # self._smooth_cache.clear()  # caching disabled
        self._layers_created = False
        has_intensity = "photon_count" in dataset.data_vars
        has_lifetime = "mean_arrival_time" in dataset.data_vars
        if not has_intensity and not has_lifetime:
            self.setVisible(False)
            return
        self.setVisible(True)
        self._create_controls(has_intensity, has_lifetime)

    # ------------------------------------------------------------------ #

    def _create_controls(self, has_intensity: bool, has_lifetime: bool):
        """
        Rebuild all child widgets inside the FLIM View group box.

        Called by update_data() each time a new dataset arrives.  Removes any
        previously created widgets, then lays out:
          col 0 — frame / sequence / channel selectors with Agg checkboxes
          col 1 — Intensity HistogramSlider + controls + mask button
          col 2 — Lifetime HistogramSlider + controls + mask button
          col 3 — Export buttons (Intensity, Lifetime, FLIM RGB)

        All signal wiring (selectors → debounced _slice, sliders → _fast_display,
        colormaps, mask buttons, smoothing, visibility) is established here using
        closures that capture the local helpers defined in this method.

        Inner helpers (not accessible outside this method):
          _make_lut(cmap_name)      — build 256×3 uint8 LUT (for export + RGB layer)
          _get_sel_key()            — return (sel dict, dims_to_sum list) from selectors
          _get_raw_arrays(sel, sums)— isel + optional aggregate (memoised only when
                                      self._use_sum_cache is True; off by default)
          _get_smoothed(raw_i, raw_l)— apply smoothing kernels (recomputed each call;
                                      caching currently disabled)
          _display_data()           — full layer update (data + contrast + colormap)
          _fast_display()           — contrast-only update for slider drag
          _slice()                  — re-extract arrays, update histograms, call _display_data
          _update_colormaps()       — push colormap changes to napari layers + histogram
          _visibility_changed()     — toggle layer visibility when ON checkboxes change
          _next_layer_name(base)    — return a unique layer name (base, base [1], …)
          _create_intensity_mask()  — add Labels layer from intensity red-slider range
          _create_lifetime_mask()   — add Labels layer from lifetime red-slider range
        """
        while self.view_layout.count():
            item = self.view_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        ds = self.dataset
        n_frames = ds.sizes.get("frame", 1)
        n_sequences = ds.sizes.get("sequence", 1)
        n_channels = ds.sizes.get("channel", 1)
        instrument_params = ds.attrs.get("instrument_params", {})
        tcspc_res_ns = instrument_params.get("tcspc_resolution_ns", 1.0)
        lifetime_unit = instrument_params.get("resolution_unit", "ch")
        self._lifetime_unit = lifetime_unit  # for the RGB export sidecar

        grid = QGridLayout()
        grid.setSpacing(2)
        # col 0: selectors (narrow), cols 1-2: histogram groups (wide), col 3: export (medium)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 3)
        grid.setColumnStretch(3, 1)
        self.view_layout.addLayout(grid)

        # ---- Col 0: slicing selectors ----
        sel_group = QGroupBox("Dims")
        apply_style(sel_group, S.GROUP_PRIMARY)
        sel_group.setFlat(True)
        sf = QGridLayout(sel_group)
        sf.setVerticalSpacing(1)

        frame_sel = QSpinBox()
        frame_sel.setRange(0, max(0, n_frames - 1))
        frame_sel.setMaximumWidth(50)
        sum_frames = QCheckBox("∑")
        sum_frames.setEnabled(n_frames > 1)
        sum_frames.setToolTip("Sum all frames")
        seq_sel = QSpinBox()
        seq_sel.setRange(0, max(0, n_sequences - 1))
        seq_sel.setMaximumWidth(50)
        sum_seqs = QCheckBox("∑")
        sum_seqs.setEnabled(n_sequences > 1)
        sum_seqs.setToolTip("Sum all sequences")
        chan_sel = QSpinBox()
        chan_sel.setRange(0, max(0, n_channels - 1))
        chan_sel.setMaximumWidth(50)
        sum_chans = QCheckBox("∑")
        sum_chans.setEnabled(n_channels > 1)
        sum_chans.setToolTip("Sum all detector channels")

        for row, (lbl, spin, chk) in enumerate(
            [
                ("Frame:", frame_sel, sum_frames),
                ("Sequence:", seq_sel, sum_seqs),
                ("Detector:", chan_sel, sum_chans),
            ]
        ):
            sf.addWidget(QLabel(lbl), row, 0)
            sf.addWidget(spin, row, 1)
            sf.addWidget(chk, row, 2)

        grid.addWidget(sel_group, 0, 0)
        self.selectors = {
            "frame": frame_sel,
            "sequence": seq_sel,
            "channel": chan_sel,
            "sum_frames": sum_frames,
            "sum_sequences": sum_seqs,
            "sum_channels": sum_chans,
        }

        # ---- Col 1: Intensity ----
        int_group = QGroupBox("Intensity")
        apply_style(int_group, S.GROUP_PRIMARY)
        int_group.setEnabled(has_intensity)
        ig = QHBoxLayout(int_group)
        ig.setSpacing(1)
        ig.setContentsMargins(4, 2, 4, 2)

        self.intensity_slider = HistogramSlider(
            integer_mode=True, canvas_height=_CANVAS_H
        )
        self.intensity_slider.setMaximumWidth(_HIST_MAX_W)
        self.intensity_slider.setToolTip(
            "Cyan slider: display contrast range\n"
            "Red slider: intensity threshold for mask creation"
        )
        ig.addWidget(self.intensity_slider, stretch=3)

        self.intensity_slider.set_name("Counts")

        int_ctrl = QVBoxLayout()
        int_ctrl.setSpacing(3)
        int_ctrl.setContentsMargins(6, 4, 6, 0)

        int_row1 = QHBoxLayout()
        int_row1.setSpacing(3)
        self.show_intensity = QCheckBox("Show")
        self.show_intensity.setChecked(True)
        self.show_intensity.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        int_row1.addWidget(self.show_intensity)
        # int_row1.addSpacing(30)
        int_row1.addStretch()
        self.smooth_int_check = QCheckBox("Smooth")
        self.smooth_int_check.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft
        )
        int_row1.addWidget(self.smooth_int_check)
        self.smooth_int_spin = QSpinBox()
        self.smooth_int_spin.setRange(3, 49)
        self.smooth_int_spin.setSingleStep(2)  # odd kernel sizes only
        self.smooth_int_spin.setValue(3)
        self.smooth_int_spin.setEnabled(False)
        self.smooth_int_check.toggled.connect(self.smooth_int_spin.setEnabled)
        int_row1.addWidget(self.smooth_int_spin)
        int_ctrl.addLayout(int_row1)

        int_row2 = QHBoxLayout()
        self.int_colormap_lbl = QLabel("Colormap:")
        int_row2.addWidget(self.int_colormap_lbl)
        self.int_colormap = QComboBox()
        self.int_colormap.addItems(INTENSITY_COLORMAPS)
        int_row2.addWidget(self.int_colormap)
        int_ctrl.addLayout(int_row2)

        int_contrast = QHBoxLayout()
        int_contrast.setSpacing(2)
        self.int_contrast_lbl = QLabel("Contrast:")
        int_contrast.addWidget(self.int_contrast_lbl)
        self.int_auto_btn = QPushButton("Auto")
        self.int_auto_btn.setToolTip("Set contrast to [p2, p98]")
        int_contrast.addWidget(self.int_auto_btn)
        self.int_minmax_btn = QPushButton("Min-Max")
        self.int_minmax_btn.setToolTip("Set contrast to full data range")
        int_contrast.addWidget(self.int_minmax_btn)
        int_ctrl.addLayout(int_contrast)

        self.int_mask_btn = QPushButton("→ Generate Int. Mask")
        # self.int_mask_btn.setStyleSheet(S.BTN_DANGER)
        self.int_mask_btn.setToolTip(
            "Create a new Labels layer from pixels within the red slider range."
        )
        self.int_mask_btn.setEnabled(False)
        int_ctrl.addWidget(self.int_mask_btn)
        int_ctrl.addStretch()
        ig.addLayout(int_ctrl)

        grid.addWidget(int_group, 0, 1)

        # ---- Col 2: Lifetime ----
        lt_group = QGroupBox(f"Lifetime ({lifetime_unit})")
        apply_style(lt_group, S.GROUP_PRIMARY)
        lt_group.setEnabled(has_lifetime)
        lg = QHBoxLayout(lt_group)
        lg.setSpacing(1)
        lg.setContentsMargins(4, 2, 4, 2)

        self.lifetime_slider = HistogramSlider(
            integer_mode=False, canvas_height=_CANVAS_H
        )
        self.lifetime_slider.setMaximumWidth(_HIST_MAX_W)
        self.lifetime_slider.setToolTip(
            "Cyan slider: display contrast range\n"
            "Red slider: lifetime threshold for mask creation"
        )
        lg.addWidget(self.lifetime_slider, stretch=3)

        self.lifetime_slider.set_name("Counts")

        lt_ctrl = QVBoxLayout()
        lt_ctrl.setSpacing(3)
        lt_ctrl.setContentsMargins(6, 4, 6, 0)

        lt_row1 = QHBoxLayout()
        lt_row1.setSpacing(3)
        self.show_lifetime = QCheckBox("Show")
        self.show_lifetime.setChecked(True)
        self.show_lifetime.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        lt_row1.addWidget(self.show_lifetime)
        self.smooth_lt_check = QCheckBox("Smooth")
        self.smooth_lt_check.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        lt_row1.addWidget(self.smooth_lt_check)
        self.smooth_lt_spin = QSpinBox()
        self.smooth_lt_spin.setRange(3, 49)
        self.smooth_lt_spin.setSingleStep(2)  # odd kernel sizes only
        self.smooth_lt_spin.setValue(3)
        self.smooth_lt_spin.setEnabled(False)
        self.smooth_lt_check.toggled.connect(self.smooth_lt_spin.setEnabled)
        lt_row1.addWidget(self.smooth_lt_spin)
        lt_ctrl.addLayout(lt_row1)

        lt_row2 = QHBoxLayout()
        lt_row2.setSpacing(2)
        self.lt_colormap_lbl = QLabel("Colormap:")
        lt_row2.addWidget(self.lt_colormap_lbl)
        self.lt_colormap = QComboBox()
        self.lt_colormap.addItems(LIFETIME_COLORMAPS)
        lt_row2.addWidget(self.lt_colormap)
        lt_ctrl.addLayout(lt_row2)

        lt_contrast = QHBoxLayout()
        lt_contrast.setSpacing(2)
        self.lt_text = QLabel("Contrast:")
        lt_contrast.addWidget(self.lt_text)
        self.lt_auto_btn = QPushButton("Auto")
        self.lt_auto_btn.setToolTip("Set contrast to [p2, p98]")
        lt_contrast.addWidget(self.lt_auto_btn)
        self.lt_minmax_btn = QPushButton("Min-Max")
        self.lt_minmax_btn.setToolTip("Set contrast to full data range")
        lt_contrast.addWidget(self.lt_minmax_btn)
        lt_ctrl.addLayout(lt_contrast)

        self.lt_mask_btn = QPushButton("→ Generate Lt. Mask")
        # self.lt_mask_btn.setStyleSheet(S.BTN_DANGER)
        self.lt_mask_btn.setToolTip(
            "Create a new Labels layer from pixels within the red slider range."
        )
        self.lt_mask_btn.setEnabled(False)
        lt_ctrl.addWidget(self.lt_mask_btn)
        lt_ctrl.addStretch()
        lg.addLayout(lt_ctrl)

        grid.addWidget(lt_group, 0, 2)

        # ---- Col 3: Export ----
        exp_group = QGroupBox("Export")
        apply_style(exp_group, S.GROUP_PRIMARY)
        el = QVBoxLayout(exp_group)
        el.setSpacing(8)
        el.setContentsMargins(4, 4, 4, 2)

        has_flim = has_intensity and has_lifetime

        # Row 1: one checkbox per export type
        chk_row = QHBoxLayout()
        chk_row.setSpacing(2)
        # chk_row.setContentsMargins(6, 0, 6, 0)
        self.exp_int_chk = QCheckBox("Intensity")
        self.exp_int_chk.setEnabled(has_intensity)
        self.exp_int_chk.setChecked(has_intensity)
        self.exp_lt_chk = QCheckBox("Lifetime")
        self.exp_lt_chk.setEnabled(has_lifetime)
        self.exp_lt_chk.setChecked(has_lifetime)
        self.exp_flim_chk = QCheckBox("FLIM RGB")
        self.exp_flim_chk.setEnabled(has_flim)
        self.exp_flim_chk.setChecked(has_flim)
        for chk in (self.exp_int_chk, self.exp_lt_chk, self.exp_flim_chk):
            chk_row.addWidget(chk)
        el.addLayout(chk_row)

        # Row 2: which slice is on screen — i.e. exactly what Save writes.
        # (Stack export will live in the Batch tab; nothing to choose here.)
        self._view_lbl = QLabel()
        self._view_lbl.setStyleSheet(S.STATUS)
        self._view_lbl.setWordWrap(True)
        self._view_lbl.setToolTip("The slice / settings currently displayed.")
        el.addWidget(self._view_lbl)

        # Shape info — shown below Stack radio (relevant for stack context).
        # Letters are the dim initials, except `channel` → D, so it reads as
        # the detector axis like everywhere else in the UI (D0, D1, …).
        _shape_letters = {
            "frame": "F",
            "sequence": "S",
            "channel": "D",
            "line": "Y",
            "pixel": "X",
        }
        _shape_parts = [
            f"{letter}: {ds.sizes[dim]}"
            for dim, letter in _shape_letters.items()
            if dim in ds.sizes
        ]
        shape_lbl = QLabel("Dataset = " + "  ".join(_shape_parts))
        shape_lbl.setStyleSheet(S.STATUS)
        shape_lbl.setWordWrap(True)
        el.addWidget(shape_lbl)

        # Row 3: Save + "→ RGB FLIM layer" side by side.
        self.export_save_btn = QPushButton("Save")
        self.export_save_btn.setEnabled(False)
        self.export_save_btn.clicked.connect(self._on_export_save)

        # Add the current FLIM RGB composite as a static napari layer (same
        # image as the RGB export, but kept in the viewer).
        self.gen_flim_btn = QPushButton("→ RGB FLIM layer")
        self.gen_flim_btn.setEnabled(has_flim)
        self.gen_flim_btn.setToolTip(
            "Add the current FLIM RGB composite (current view, contrast,\n"
            "colormap and smoothing) as a static napari image layer."
        )
        self.gen_flim_btn.clicked.connect(self._generate_flim_layer)

        save_row = QHBoxLayout()
        save_row.setSpacing(2)
        save_row.addWidget(self.export_save_btn)
        save_row.addWidget(self.gen_flim_btn)
        el.addLayout(save_row)
        # Pack rows to the top (like int_ctrl/lt_ctrl) so the extra height from
        # the taller Intensity/Lifetime boxes doesn't spread the rows apart —
        # keeps the inter-row spacing matching the other columns.
        el.addStretch()

        grid.addWidget(exp_group, 0, 3)

        self.int_auto_btn.clicked.connect(
            self.intensity_slider.set_auto_contrast
        )
        self.int_minmax_btn.clicked.connect(self.intensity_slider.set_min_max)
        self.lt_auto_btn.clicked.connect(
            self.lifetime_slider.set_auto_contrast
        )
        self.lt_minmax_btn.clicked.connect(self.lifetime_slider.set_min_max)

        def _sync_int_histogram_colormap():
            """In FLIM mode intensity histogram always shows gray; otherwise follow combo."""
            show_i = self.show_intensity.isChecked() and has_intensity
            show_l = self.show_lifetime.isChecked() and has_lifetime
            flim_mode = show_i and show_l
            self.int_colormap.setEnabled(not flim_mode)
            cmap = (
                colormaps["gray"]
                if flim_mode
                else colormaps[self.int_colormap.currentText()]
            )
            self.intensity_slider.set_colormap(cmap)

        self.show_intensity.toggled.connect(
            lambda _: _sync_int_histogram_colormap()
        )
        self.show_lifetime.toggled.connect(
            lambda _: _sync_int_histogram_colormap()
        )
        self.int_colormap.currentTextChanged.connect(
            lambda _: _sync_int_histogram_colormap()
        )
        _sync_int_histogram_colormap()

        # ------------------------------------------------------------------ #
        # Helpers                                                              #
        # ------------------------------------------------------------------ #

        def _set_visible(name: str, value: bool):
            if name in self.viewer.layers:
                self.viewer.layers[name].visible = value

        def _make_lut(cmap_name: str) -> np.ndarray:
            return colormap_to_lut(cmap_name)

        def _int_cmap():
            # In FLIM mode (both layers shown) intensity MUST be gray so the
            # multiplicative composite = gray_intensity x lifetime_colour.
            # Otherwise honour the user's intensity colormap combo.
            both = (
                has_intensity
                and has_lifetime
                and self.show_intensity.isChecked()
                and self.show_lifetime.isChecked()
            )
            return "gray" if both else self.int_colormap.currentText()

        def _get_sel_key():
            sel, dims_to_sum = {}, []
            for dim in ["frame", "sequence", "channel"]:
                sum_key = f"sum_{dim}s"
                if dim in ds.sizes and self.selectors[sum_key].isChecked():
                    dims_to_sum.append(dim)
                elif dim in ds.sizes:
                    sel[dim] = self.selectors[dim].value()
            return sel, dims_to_sum

        def _get_raw_arrays(sel, dims_to_sum):
            cache_key = (tuple(sorted(sel.items())), frozenset(dims_to_sum))
            if self._use_sum_cache and cache_key in self._sum_cache:
                return self._sum_cache[cache_key]
            sliced = ds.isel(**sel)
            final = (
                aggregate_dataset(sliced, dims_to_sum)
                if dims_to_sum
                else sliced
            )
            raw_int = (
                final["photon_count"].values.squeeze()
                if "photon_count" in final
                else None
            )
            raw_lt = (
                final["mean_arrival_time"].values.squeeze()
                if "mean_arrival_time" in final
                else None
            )
            if self._use_sum_cache:
                self._sum_cache[cache_key] = (raw_int, raw_lt)
                if len(self._sum_cache) > 64:
                    self._sum_cache.pop(next(iter(self._sum_cache)))
            return raw_int, raw_lt

        def _get_smoothed(raw_int, raw_lt):
            s_int = raw_int
            if (
                has_intensity
                and raw_int is not None
                and self.smooth_int_check.isChecked()
            ):
                k = self.smooth_int_spin.value()
                s_int = smooth_count(raw_int, size=k)
                # --- smoothing cache disabled (kept for reference) ---
                # key = ("int", id(raw_int), raw_int.shape, k)
                # if key not in self._smooth_cache:
                #     self._smooth_cache[key] = smooth_count(raw_int, size=k)
                #     if len(self._smooth_cache) > 32:
                #         self._smooth_cache.pop(next(iter(self._smooth_cache)))
                # s_int = self._smooth_cache[key]
            s_lt = raw_lt
            if (
                has_lifetime
                and raw_lt is not None
                and raw_int is not None
                and self.smooth_lt_check.isChecked()
            ):
                k = self.smooth_lt_spin.value()
                s_lt, _ = smooth_weighted(raw_lt, raw_int, size=k)
                # --- smoothing cache disabled (kept for reference) ---
                # key = ("lt", id(raw_lt), raw_lt.shape, k)
                # if key not in self._smooth_cache:
                #     result, _ = smooth_weighted(raw_lt, raw_int, size=k)
                #     self._smooth_cache[key] = result
                #     if len(self._smooth_cache) > 32:
                #         self._smooth_cache.pop(next(iter(self._smooth_cache)))
                # s_lt = self._smooth_cache[key]
            return s_int, s_lt

        # ------------------------------------------------------------------ #
        # Display                                                              #
        # ------------------------------------------------------------------ #

        def _display_data():
            try:
                ci = self._current_intensity
                cl = self._current_lifetime
                show_i = self.show_intensity.isChecked() and has_intensity
                show_l = self.show_lifetime.isChecked() and has_lifetime

                if not self._layers_created:
                    for name in ["FLIM", "Intensity", "Lifetime"]:
                        if name in self.viewer.layers:
                            self.viewer.layers.remove(name)
                    # Intensity = gray base; Lifetime sits on top with
                    # multiplicative blending, so napari composites the FLIM
                    # look on the GPU (Intensity x colormapped Lifetime).
                    if ci is not None:
                        self.viewer.add_image(
                            ci, name="Intensity", colormap="gray"
                        )
                    if cl is not None:
                        self.viewer.add_image(
                            cl,
                            name="Lifetime",
                            colormap="rainbow",
                            blending="multiplicative",
                        )
                    self._layers_created = True

                # Both layers always carry their own data / contrast / colormap;
                # the ON checkboxes just toggle visibility. Both visible = FLIM.
                _set_visible("Intensity", show_i)
                if "Intensity" in self.viewer.layers and ci is not None:
                    layer = self.viewer.layers["Intensity"]
                    layer.data = ci
                    layer.contrast_limits = self.intensity_slider.value()
                    layer.colormap = _int_cmap()

                _set_visible("Lifetime", show_l)
                if "Lifetime" in self.viewer.layers and cl is not None:
                    layer = self.viewer.layers["Lifetime"]
                    layer.data = cl
                    layer.contrast_limits = self.lifetime_slider.value()
                    layer.colormap = self.lt_colormap.currentText()

                # enable Save button whenever any cached data is available
                self.export_save_btn.setEnabled(
                    ci is not None or cl is not None
                )
            except Exception:
                traceback.print_exc()

        def _fast_display():
            """Slider drag — update contrast only; napari re-composites on GPU."""
            try:
                if (
                    has_intensity
                    and "Intensity" in self.viewer.layers
                    and self._current_intensity is not None
                ):
                    self.viewer.layers["Intensity"].contrast_limits = (
                        self.intensity_slider.value()
                    )
                if (
                    has_lifetime
                    and "Lifetime" in self.viewer.layers
                    and self._current_lifetime is not None
                ):
                    self.viewer.layers["Lifetime"].contrast_limits = (
                        self.lifetime_slider.value()
                    )
            except Exception:
                traceback.print_exc()

        _first_slice = [True]

        def _reset_and_slice():
            """Force a full contrast reset on the next slice (for smooth/aggregate changes)."""
            _first_slice[0] = True
            _slice()

        def _slice():
            try:
                sel, dims_to_sum = _get_sel_key()
                raw_int, raw_lt = _get_raw_arrays(sel, dims_to_sum)
                s_int, s_lt = _get_smoothed(raw_int, raw_lt)
                self._current_intensity = (
                    np.atleast_2d(s_int) if s_int is not None else None
                )
                self._current_lifetime = (
                    np.atleast_2d(s_lt) * tcspc_res_ns
                    if s_lt is not None
                    else None
                )
                update_slider = (
                    self.intensity_slider.update_data
                    if _first_slice[0]
                    else self.intensity_slider.update_data_keep_range
                )
                update_lt_slider = (
                    self.lifetime_slider.update_data
                    if _first_slice[0]
                    else self.lifetime_slider.update_data_keep_range
                )
                if self._current_intensity is not None:
                    update_slider(self._current_intensity)
                if self._current_lifetime is not None:
                    update_lt_slider(self._current_lifetime)
                _first_slice[0] = False
                _display_data()
                # Notify phasor/decay panels about current view settings
                self.view_changed.emit(
                    {
                        "frame": self.selectors["frame"].value(),
                        "sequence": self.selectors["sequence"].value(),
                        "channel": self.selectors["channel"].value(),
                        "sum_frames": self.selectors["sum_frames"].isChecked(),
                        "sum_sequences": self.selectors[
                            "sum_sequences"
                        ].isChecked(),
                        "sum_channels": self.selectors[
                            "sum_channels"
                        ].isChecked(),
                    }
                )
            except Exception:
                traceback.print_exc()

        def _update_colormaps():
            try:
                if has_intensity:
                    self.intensity_slider.set_colormap(
                        colormaps[self.int_colormap.currentText()]
                    )
                    if "Intensity" in self.viewer.layers:
                        self.viewer.layers["Intensity"].colormap = _int_cmap()
                if has_lifetime:
                    self.lifetime_slider.set_colormap(
                        colormaps[self.lt_colormap.currentText()]
                    )
                    # LUT still drives the RGB export + "→ RGB FLIM layer".
                    self._lut = _make_lut(self.lt_colormap.currentText())
                    if "Lifetime" in self.viewer.layers:
                        self.viewer.layers["Lifetime"].colormap = (
                            self.lt_colormap.currentText()
                        )
            except Exception:
                traceback.print_exc()

        def _visibility_changed():
            try:
                _set_visible(
                    "Intensity",
                    self.show_intensity.isChecked() and has_intensity,
                )
                _set_visible(
                    "Lifetime",
                    self.show_lifetime.isChecked() and has_lifetime,
                )
                # Entering/leaving FLIM mode flips intensity gray <-> combo.
                if "Intensity" in self.viewer.layers:
                    self.viewer.layers["Intensity"].colormap = _int_cmap()
            except Exception:
                traceback.print_exc()

        def _next_layer_name(base: str) -> str:
            existing = {layer.name for layer in self.viewer.layers}
            if base not in existing:
                return base
            i = 1
            while f"{base} [{i}]" in existing:
                i += 1
            return f"{base} [{i}]"

        def _create_intensity_mask():
            ci = self._current_intensity
            if ci is None:
                return
            lo, hi = self.intensity_slider.mask_value()
            mask = np.where((ci >= lo) & (ci <= hi), 1, 0).astype(np.int32)
            self.viewer.add_labels(
                mask, name=_next_layer_name("Intensity Mask")
            )

        def _create_lifetime_mask():
            cl = self._current_lifetime
            if cl is None:
                return
            lo, hi = self.lifetime_slider.mask_value()
            mask = np.where((cl >= lo) & (cl <= hi), 1, 0).astype(np.int32)
            self.viewer.add_labels(
                mask, name=_next_layer_name("Lifetime Mask")
            )

        # ---- wire signals ----
        # Spinboxes → debounce → _slice (sticky contrast on navigation)
        # Agg checkboxes → _reset_and_slice (contrast resets when aggregation changes)
        # The view label is pure string formatting, so it updates immediately
        # rather than waiting on the slice debounce.
        for w in self.selectors.values():
            if isinstance(w, QSpinBox):
                w.valueChanged.connect(self._debounce.start)
                w.valueChanged.connect(self._update_view_label)
            if isinstance(w, QCheckBox):
                w.toggled.connect(_reset_and_slice)
                w.toggled.connect(self._update_view_label)
        self._debounce.timeout.connect(_slice)

        self.smooth_int_check.toggled.connect(_reset_and_slice)
        self.smooth_int_spin.valueChanged.connect(_reset_and_slice)
        self.show_intensity.toggled.connect(_visibility_changed)
        self.int_colormap.currentTextChanged.connect(_update_colormaps)
        self.intensity_slider.valueChanged.connect(_fast_display)
        self.intensity_slider.sliderReleased.connect(_fast_display)
        if has_intensity:
            self.int_mask_btn.setEnabled(True)
            self.int_mask_btn.clicked.connect(_create_intensity_mask)
        if has_lifetime:
            self.lt_mask_btn.setEnabled(True)
            self.lt_mask_btn.clicked.connect(_create_lifetime_mask)

        if has_lifetime:
            self.smooth_lt_check.toggled.connect(_reset_and_slice)
            self.smooth_lt_spin.valueChanged.connect(_reset_and_slice)
            self.show_lifetime.toggled.connect(_visibility_changed)
            self.lt_colormap.currentTextChanged.connect(_update_colormaps)
            self.lifetime_slider.valueChanged.connect(_fast_display)
            self.lifetime_slider.sliderReleased.connect(_fast_display)

        self._lut = (
            _make_lut(self.lt_colormap.currentText()) if has_lifetime else None
        )
        _slice()
        _update_colormaps()
        self._update_view_label()

    def _update_view_label(self):
        """Refresh the current-view line, e.g. ``View:  f0  s0  d1  l  p``.

        One token per dimension: the selected index, or ``A`` when that
        dimension is summed. The spatial axes carry no index — they are always
        shown whole.
        """
        ds = self.dataset
        if ds is None or not self.selectors:
            return
        parts = []
        for dim, letter in (
            ("frame", "F"),
            ("sequence", "S"),
            ("channel", "D"),
        ):
            if dim not in ds.sizes:
                continue
            if self.selectors[f"sum_{dim}s"].isChecked():
                parts.append(f"{letter}: ∑")
            else:
                parts.append(f"{letter}: {self.selectors[dim].value()}")
        self._view_lbl.setText("Current view =  " + "  ".join(parts))

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def _on_export_save(self):
        """Unified Save handler — asks for base name/directory once, saves all checked types."""
        do_int = self.exp_int_chk.isChecked() and self.exp_int_chk.isEnabled()
        do_lt = self.exp_lt_chk.isChecked() and self.exp_lt_chk.isEnabled()
        do_flim = (
            self.exp_flim_chk.isChecked() and self.exp_flim_chk.isEnabled()
        )
        if not (do_int or do_lt or do_flim):
            return

        stem = Path(self.dataset.attrs.get("source_filename", "export")).stem
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save — choose base name and location", stem, "All Files (*)"
        )
        if not path_str:
            return

        # Taken literally — whatever was typed is the base name, so a dot in it
        # survives ("scan_44.56"). Type an extension and you get it twice; the
        # per-type suffix below is always appended.
        base = Path(path_str)
        try:
            if do_int:
                self._save_intensity(base.parent / (base.name + "_int.tif"))
            if do_lt:
                self._save_lifetime(base.parent / (base.name + "_lt.tif"))
            if do_flim:
                self._save_flim(base.parent / (base.name + "_flim.png"))
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _generate_flim_layer(self):
        """Add the current FLIM RGB composite as a static napari image layer.

        Uses the same core.flim_rgb as the file export, so this layer matches an
        exported ``_flim.png`` (current view / contrast / colormap / smoothing).
        """
        ci, cl = self._current_intensity, self._current_lifetime
        if ci is None or cl is None or self._lut is None:
            return
        rgb = flim_rgb(
            ci,
            cl,
            self._lut,
            self.lifetime_slider.value(),
            self.intensity_slider.value(),
        )
        existing = {layer.name for layer in self.viewer.layers}
        name, i = "FLIM RGB", 1
        while name in existing:
            name = f"FLIM RGB [{i}]"
            i += 1
        self.viewer.add_image(
            rgb,
            rgb=True,
            name=name,
            metadata={"flim": self._flim_export_info()},
        )

    def _save_intensity(self, path):
        """Write cached intensity as actual photon counts (uint32 TIFF).

        Not rescaled — pixel values are the real counts (rounded when the view
        is smoothed/aggregated), so the TIFF stays quantitative.
        """
        from skimage.io import imsave

        counts = np.rint(self._current_intensity).astype(np.uint32)
        imsave(str(path), counts, check_contrast=False)

    def _save_lifetime(self, path):
        """Write cached lifetime as float32 TIFF (values in ns)."""
        from skimage.io import imsave

        imsave(
            str(path),
            self._current_lifetime.astype(np.float32),
            check_contrast=False,
        )

    def _save_flim(self, path):
        """Write FLIM RGB composite as uint8 PNG/TIFF using current LUT and contrast."""
        from skimage.io import imsave

        # Use the exact same compositing as the interactive display so the
        # exported image matches what is on screen (incl. NaN handling).
        rgb_f32 = flim_rgb(
            self._current_intensity,
            self._current_lifetime,
            self._lut,
            self.lifetime_slider.value(),
            self.intensity_slider.value(),
        )
        imsave(
            str(path),
            (rgb_f32 * 255).clip(0, 255).astype(np.uint8),
            check_contrast=False,
        )
        # Sidecar text so the RGB is interpretable (colormap + contrast ranges).
        Path(path).with_suffix(".txt").write_text(
            self._flim_export_info(), encoding="utf-8"
        )

    def _flim_export_info(self) -> str:
        """Human-readable mapping baked into the FLIM RGB (for the sidecar)."""
        return flim_export_info(
            self.lt_colormap.currentText(),
            self.lifetime_slider.value(),
            self.intensity_slider.value(),
            getattr(self, "_lifetime_unit", "ch"),
        )
