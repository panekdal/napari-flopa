"""
Fast histogram + dual range slider widget.

The histogram canvas shows two overlaid range indicators:
  - View range : cyan  dashed lines  (top slider, for display contrast)
  - Mask range : red   dashed lines  (bottom slider, for mask creation)

"""

import numpy as np
from matplotlib import colormaps
from qtpy.QtCore import QRect, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QDoubleRangeSlider, QRangeSlider

from napari_flopa.ui.style import S


class _HistogramCanvas(QWidget):
    """Pure QPainter histogram bar chart with view (cyan) and mask (red) range markers."""

    def __init__(self, canvas_height=80, parent=None):
        super().__init__(parent)
        self.setFixedHeight(canvas_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._counts = np.array([])
        self._edges = np.array([])
        self._data_min = 0.0
        self._data_max = 1.0
        self._display_max = 1.0

        self._lo = 0.0
        self._hi = 1.0
        self._mask_lo = 0.0
        self._mask_hi = 1.0

        self._colormap = colormaps["gray"]
        self._lut = self._make_lut(colormaps["gray"])
        self._name = ""

    @staticmethod
    def _make_lut(cmap) -> np.ndarray:
        return (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

    def set_name(self, name: str):
        self._name = name
        self.update()

    def set_colormap(self, cmap):
        self._colormap = cmap
        self._lut = self._make_lut(cmap)
        self.update()

    def set_data(self, data: np.ndarray):
        if data is None or data.size == 0:
            self._counts = np.array([])
            self.update()
            return
        n_bins = min(128, max(32, data.size // 100))
        counts, edges = np.histogram(data, bins=n_bins)
        self._counts = np.log1p(counts.astype(np.float32))
        self._edges = edges.astype(np.float32)
        self._data_min = float(edges[0])
        self._data_max = float(edges[-1])
        span = self._data_max - self._data_min
        self._display_max = self._data_max + max(span * 0.04, 1e-6)
        self.update()

    def set_range(self, lo: float, hi: float):
        self._lo = lo
        self._hi = hi
        self.update()

    def set_mask_range(self, lo: float, hi: float):
        self._mask_lo = lo
        self._mask_hi = hi
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        w, h = self.width(), self.height()
        pad_b = 4
        pad_l = _HIST_PAD_L  # left gutter = y-axis / left slider handle
        pad_r = _HIST_PAD_R  # right padding = right slider handle

        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        # ── Y-axis labels (drawn first so they sit in the gutter) ──────────
        ax_color = QColor(150, 150, 150, 160)
        painter.setPen(ax_color)
        font = painter.font()
        font.setPointSize(6)
        painter.setFont(font)
        fm = painter.fontMetrics()

        if self._counts.size > 0 and self._data_max > self._data_min:
            max_count = self._counts.max() or 1.0
            max_label = _compact_count(int(np.expm1(max_count)))
            # "0" at bottom-left, max count at top-left — right-aligned in gutter
            painter.drawText(
                QRect(0, h - pad_b - fm.height(), pad_l - 3, fm.height()),
                Qt.AlignRight | Qt.AlignVCenter,
                "0",
            )
            painter.drawText(
                QRect(0, 1, pad_l - 3, fm.height()),
                Qt.AlignRight | Qt.AlignVCenter,
                max_label,
            )

        # Rotated name label centred vertically in the gutter
        if self._name:
            font.setPointSize(7)
            painter.setFont(font)
            painter.setPen(QColor(180, 180, 180, 140))
            painter.save()
            painter.translate(fm.height() + 2, h // 2)
            painter.rotate(-90)
            painter.drawText(
                -painter.fontMetrics().horizontalAdvance(self._name) // 2,
                0,
                self._name,
            )
            painter.restore()

        if self._counts.size == 0 or self._data_max <= self._data_min:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(
                QRect(pad_l, 0, w - pad_l, h), Qt.AlignCenter, "No data"
            )
            painter.end()
            return

        # ── Histogram bars (x mapped onto [pad_l, w - pad_r]) ─────────────
        pad_t = 6  # top buffer so the tallest bar never touches the ceiling
        bar_w = w - pad_l - pad_r
        bar_area_h = h - pad_b - pad_t  # usable vertical space for bars
        disp_range = self._display_max - self._data_min
        data_range = self._data_max - self._data_min
        max_count = self._counts.max() or 1.0

        # Subtle horizontal dashed line at the max-count level
        painter.setPen(QPen(QColor(150, 150, 150, 55), 1, Qt.DashLine))
        painter.drawLine(pad_l, pad_t, w - pad_r, pad_t)

        for i, count in enumerate(self._counts):
            x0, x1 = self._edges[i], self._edges[i + 1]
            px0 = pad_l + int((x0 - self._data_min) / disp_range * bar_w)
            px1 = pad_l + int((x1 - self._data_min) / disp_range * bar_w)
            bar_h = int(count / max_count * bar_area_h)

            centre = (x0 + x1) / 2.0
            if centre < self._lo or centre > self._hi:
                lut_idx = int((centre - self._data_min) / data_range * 255)
                lut_idx = max(0, min(255, lut_idx))
                r = int(self._lut[lut_idx, 0]) // 4
                g = int(self._lut[lut_idx, 1]) // 4
                b = int(self._lut[lut_idx, 2]) // 4
            else:
                view_range = (
                    self._hi - self._lo if self._hi > self._lo else 1.0
                )
                lut_idx = int((centre - self._lo) / view_range * 255)
                lut_idx = max(0, min(255, lut_idx))
                r = int(self._lut[lut_idx, 0])
                g = int(self._lut[lut_idx, 1])
                b = int(self._lut[lut_idx, 2])

            painter.fillRect(
                px0,
                h - pad_b - bar_h,
                max(1, px1 - px0),
                bar_h,
                QColor(r, g, b),
            )

        # View range markers — cyan dashed, semi-transparent
        cyan = QColor(0, 220, 220, 140)
        for val in (self._lo, self._hi):
            px = pad_l + int((val - self._data_min) / disp_range * bar_w)
            painter.setPen(QPen(cyan, 1, Qt.DashLine))
            painter.drawLine(px, pad_t, px, h - pad_b)

        # Mask range markers — red dashed, semi-transparent
        red = QColor(220, 60, 60, 140)
        for val in (self._mask_lo, self._mask_hi):
            px = pad_l + int((val - self._data_min) / disp_range * bar_w)
            painter.setPen(QPen(red, 1, Qt.DashLine))
            painter.drawLine(px, pad_t, px, h - pad_b)

        painter.end()


def _compact_count(n: int) -> str:
    """Format a count value compactly: 1_200 → '1.2k', 1_500_000 → '1.5M'."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


_HIST_PAD_L = (
    52  # left gutter = edit-box width; y-axis aligns with left slider handle
)
_HIST_PAD_R = 45  # right padding = edit-box width; x-axis end aligns with right slider handle


class HistogramSlider(QWidget):
    """
    Compound widget: QPainter histogram + two superqt range sliders.

    Top slider   (cyan handles) — view range for display contrast
    Bottom slider (red handles) — mask range for mask creation

    All four handle values are shown as editable QLineEdit fields.

    Signals:
        valueChanged(tuple)      — view slider moved
        sliderReleased(tuple)    — view slider released
        maskValueChanged(tuple)  — mask slider moved
    """

    valueChanged = Signal(tuple)
    sliderReleased = Signal(tuple)
    maskValueChanged = Signal(tuple)

    def __init__(self, integer_mode=False, canvas_height=80, parent=None):
        super().__init__(parent)
        self.integer_mode = integer_mode
        self._is_updating = False
        self._valid_data: np.ndarray = np.array([])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(0)

        # ── Histogram canvas ──────────────────────────────────────────────
        self._canvas = _HistogramCanvas(canvas_height=canvas_height)
        canvas_row = QHBoxLayout()
        canvas_row.setContentsMargins(
            0, 0, 0, 0
        )  # canvas spans full widget width
        canvas_row.addWidget(self._canvas)
        layout.addLayout(canvas_row)
        layout.addSpacing(1)

        # ── View slider row (cyan) ────────────────────────────────────────
        self._lo_edit = self._make_edit()
        self._hi_edit = self._make_edit()

        if integer_mode:
            self._slider = QRangeSlider(Qt.Horizontal)
        else:
            self._slider = QDoubleRangeSlider(Qt.Horizontal)
        self._slider.setRange(0, 1)
        self._slider.setValue((0, 1))
        self._slider.setFixedHeight(18)
        self._slider.setStyleSheet(S.SLIDER_VIEW)

        view_row = QHBoxLayout()
        view_row.setSpacing(2)
        view_row.addWidget(self._lo_edit)
        view_row.addWidget(self._slider)
        view_row.addWidget(self._hi_edit)
        layout.addLayout(view_row)
        layout.addSpacing(1)

        # ── Mask slider row (red) ─────────────────────────────────────────
        self._mask_lo_edit = self._make_edit()
        self._mask_hi_edit = self._make_edit()

        if integer_mode:
            self._mask_slider = QRangeSlider(Qt.Horizontal)
        else:
            self._mask_slider = QDoubleRangeSlider(Qt.Horizontal)
        self._mask_slider.setRange(0, 1)
        self._mask_slider.setValue((0, 1))
        self._mask_slider.setFixedHeight(18)
        self._mask_slider.setStyleSheet(S.SLIDER_MASK)

        mask_row = QHBoxLayout()
        mask_row.setSpacing(2)
        mask_row.addWidget(self._mask_lo_edit)
        mask_row.addWidget(self._mask_slider)
        mask_row.addWidget(self._mask_hi_edit)
        layout.addLayout(mask_row)

        # ── Connect signals ───────────────────────────────────────────────

        self._slider.valueChanged.connect(self._on_view_moved)
        self._slider.sliderReleased.connect(self._on_view_released)
        self._mask_slider.valueChanged.connect(self._on_mask_moved)

        self._lo_edit.editingFinished.connect(
            lambda: self._apply_edit(self._lo_edit, "view_lo")
        )
        self._hi_edit.editingFinished.connect(
            lambda: self._apply_edit(self._hi_edit, "view_hi")
        )
        self._mask_lo_edit.editingFinished.connect(
            lambda: self._apply_edit(self._mask_lo_edit, "mask_lo")
        )
        self._mask_hi_edit.editingFinished.connect(
            lambda: self._apply_edit(self._mask_hi_edit, "mask_hi")
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update_data(self, data: np.ndarray):
        """Full reset: histogram + slider positions set to [p2, p98]."""
        valid = data[np.isfinite(data)] if data is not None else np.array([])
        if valid.size == 0:
            self._canvas.set_data(None)
            return

        self._valid_data = valid
        mn, mx = float(np.min(valid)), float(np.max(valid))
        if mn >= mx:
            mx = mn + 1.0

        self._canvas.set_data(valid)

        self._is_updating = True
        self._slider.setRange(mn, mx)
        self._mask_slider.setRange(mn, mx)

        lo, hi = np.percentile(valid, [2, 98])
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            lo, hi = mn, mx
        self._slider.setValue((lo, hi))
        self._is_updating = False

        self._canvas.set_range(lo, hi)
        self._update_edit(self._lo_edit, lo)
        self._update_edit(self._hi_edit, hi)

        # Mask defaults: 1 to max (integer) or data_min to max (float)
        mask_lo = 1 if self.integer_mode else mn
        self._mask_slider.setValue((mask_lo, mx))
        self._canvas.set_mask_range(mask_lo, mx)
        self._update_edit(self._mask_lo_edit, mask_lo)
        self._update_edit(self._mask_hi_edit, mx)

    def update_data_keep_range(self, data: np.ndarray):
        """Update histogram but preserve current slider positions, clamping to new range."""
        valid = data[np.isfinite(data)] if data is not None else np.array([])
        if valid.size == 0:
            self._canvas.set_data(None)
            return

        self._valid_data = valid
        mn, mx = float(np.min(valid)), float(np.max(valid))
        if mn >= mx:
            mx = mn + 1.0

        self._canvas.set_data(valid)

        cur_lo, cur_hi = float(self._slider.value()[0]), float(
            self._slider.value()[1]
        )
        new_lo = max(mn, min(cur_lo, mx))
        new_hi = max(mn, min(cur_hi, mx))
        if new_lo >= new_hi:
            new_lo, new_hi = mn, mx

        cur_mlo = float(self._mask_slider.value()[0])
        cur_mhi = float(self._mask_slider.value()[1])
        new_mlo = max(mn, min(cur_mlo, mx))
        new_mhi = max(mn, min(cur_mhi, mx))
        if new_mlo >= new_mhi:
            new_mlo = float(1 if self.integer_mode else mn)
            new_mhi = mx

        self._is_updating = True
        self._slider.setRange(mn, mx)
        self._mask_slider.setRange(mn, mx)
        self._slider.setValue((new_lo, new_hi))
        self._mask_slider.setValue((new_mlo, new_mhi))
        self._is_updating = False

        self._canvas.set_range(new_lo, new_hi)
        self._canvas.set_mask_range(new_mlo, new_mhi)
        self._update_edit(self._lo_edit, new_lo)
        self._update_edit(self._hi_edit, new_hi)
        self._update_edit(self._mask_lo_edit, new_mlo)
        self._update_edit(self._mask_hi_edit, new_mhi)

    def set_auto_contrast(self):
        """Set contrast slider to [p2, p98] of the current data."""
        if self._valid_data.size == 0:
            return
        lo, hi = np.nanpercentile(self._valid_data, [2, 98])
        mn, mx = float(self._slider.minimum()), float(self._slider.maximum())
        lo, hi = float(max(mn, lo)), float(min(mx, hi))
        if lo >= hi:
            return
        self._is_updating = True
        self._slider.setValue((lo, hi))
        self._is_updating = False
        self._canvas.set_range(lo, hi)
        self._update_edit(self._lo_edit, lo)
        self._update_edit(self._hi_edit, hi)
        self.sliderReleased.emit(self.value())

    def set_min_max(self):
        """Set contrast slider to full data range [min, max]."""
        mn, mx = float(self._slider.minimum()), float(self._slider.maximum())
        self._is_updating = True
        self._slider.setValue((mn, mx))
        self._is_updating = False
        self._canvas.set_range(mn, mx)
        self._update_edit(self._lo_edit, mn)
        self._update_edit(self._hi_edit, mx)
        self.sliderReleased.emit(self.value())

    def set_name(self, name: str):
        self._canvas.set_name(name)

    def set_colormap(self, cmap):
        self._canvas.set_colormap(cmap)

    def value(self) -> tuple:
        lo, hi = self._slider.value()
        return (
            (int(lo), int(hi)) if self.integer_mode else (float(lo), float(hi))
        )

    def mask_value(self) -> tuple:
        lo, hi = self._mask_slider.value()
        return (
            (int(lo), int(hi)) if self.integer_mode else (float(lo), float(hi))
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _make_edit(self) -> QLineEdit:
        e = QLineEdit("0")
        e.setFixedWidth(45)
        e.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        e.setStyleSheet(S.LINE_EDIT)
        return e

    def _fmt(self, v) -> str:
        return str(int(v)) if self.integer_mode else f"{v:.3f}"

    def _update_edit(self, edit: QLineEdit, v):
        if not edit.hasFocus():
            edit.setText(self._fmt(v))

    def _apply_edit(self, edit: QLineEdit, target: str):
        try:
            v = int(edit.text()) if self.integer_mode else float(edit.text())
        except ValueError:
            # Reset to current slider value on bad input
            lo, hi = self._slider.value()
            mlo, mhi = self._mask_slider.value()
            lookup = {
                "view_lo": lo,
                "view_hi": hi,
                "mask_lo": mlo,
                "mask_hi": mhi,
            }
            edit.setText(self._fmt(lookup[target]))
            return

        mn = self._slider.minimum()
        mx = self._slider.maximum()
        lo, hi = self._slider.value()
        mlo, mhi = self._mask_slider.value()

        if target == "view_lo":
            v = max(mn, min(v, hi))
            self._slider.setValue((v, hi))
        elif target == "view_hi":
            v = max(lo, min(v, mx))
            self._slider.setValue((lo, v))
        elif target == "mask_lo":
            v = max(mn, min(v, mhi))
            self._mask_slider.setValue((v, mhi))
        elif target == "mask_hi":
            v = max(mlo, min(v, mx))
            self._mask_slider.setValue((mlo, v))

    def _on_view_moved(self):
        if self._is_updating:
            return
        lo, hi = self._slider.value()
        self._canvas.set_range(lo, hi)
        self._update_edit(self._lo_edit, lo)
        self._update_edit(self._hi_edit, hi)
        self.valueChanged.emit(self.value())

    def _on_view_released(self):
        if self._is_updating:
            return
        self.sliderReleased.emit(self.value())

    def _on_mask_moved(self):
        if self._is_updating:
            return
        lo, hi = self._mask_slider.value()
        self._canvas.set_mask_range(lo, hi)
        self._update_edit(self._mask_lo_edit, lo)
        self._update_edit(self._mask_hi_edit, hi)
        self.maskValueChanged.emit(self.mask_value())
