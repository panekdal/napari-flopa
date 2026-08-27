"""
Dependency-free line plot for dense traces with a draggable time selector.

QPainter rather than matplotlib on purpose: a trace can hold hundreds of
thousands of bins and the interaction (click to place a selector, drag to pick a
range) is custom. The analytical tabs stay on matplotlib, which brings axes,
log scales, colormaps and export that this widget deliberately does not.

Signals:
    selector_changed(float)              — selector moved to this time
    selection_range_changed(float, float) — a drag finished over this range
"""

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import QWidget

from napari_flopa.ui.style import MPL, C

_MARKER_STYLE = {
    "line_start": ("line_start_times", MPL.MARKER_LINE_START),
    "line_stop": ("line_stop_times", MPL.MARKER_LINE_STOP),
    "frame_start": ("frame_start_times", MPL.MARKER_FRAME),
}

_PLOT_MARGINS = (74, 26, 25, 40)  # left, top, right, bottom


class LinePlotWidget(QWidget):
    """Time trace plot: one line per detector plus marker overlays."""

    selector_changed = Signal(float)
    selection_range_changed = Signal(float, float)
    selection_cleared = Signal()  # a plain click discarded the shaded range

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series: list[tuple] = []
        self._marker_series: dict[str, tuple] = {}
        self._selector_time: float | None = None
        self._dragging = False
        self._drag_start_time: float | None = None
        self._drag_end_time: float | None = None
        self._selection: tuple[float, float] | None = None
        self._x_min: float | None = None
        self._x_max: float | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(320)

    def set_values(
        self,
        dataset,
        *,
        start_time: float = 0.0,
        stop_time: float = 10.0,
        visible_channels=None,
        sum_selected: bool = False,
        marker_visibility: dict | None = None,
    ):
        """Show ``photon_count`` from a trace dataset (None clears the plot)."""
        self._series = []
        self._marker_series = {}
        self._selector_time = None
        self._x_min = float(start_time)
        self._x_max = float(stop_time)

        if dataset is None or "photon_count" not in getattr(
            dataset, "data_vars", {}
        ):
            self.update()
            return

        photon_count = dataset["photon_count"]
        time_axis = np.asarray(photon_count.coords["time"].values)
        counts = np.asarray(photon_count.values)
        if counts.ndim == 1:
            counts = counts.reshape(-1, 1)
        channel_ids = np.asarray(photon_count.coords["channel"].values)

        wanted = (
            {int(c) for c in channel_ids}
            if visible_channels is None
            else {int(c) for c in visible_channels}
        )
        for index in range(counts.shape[1]):
            channel_id = (
                int(channel_ids[index]) if index < len(channel_ids) else index
            )
            if channel_id not in wanted:
                continue
            self._series.append(
                (
                    time_axis,
                    counts[:, index],
                    QColor(C.SERIES[index % len(C.SERIES)]),
                    f"Detector {channel_id}",
                )
            )

        if sum_selected and self._series:
            summed = np.sum(
                np.asarray([s[1] for s in self._series], dtype=float), axis=0
            )
            self._series.append(
                (self._series[0][0], summed, QColor(C.TEXT), "Sum")
            )

        for name, (var_name, colour) in _MARKER_STYLE.items():
            if var_name not in dataset.data_vars:
                continue
            if marker_visibility is not None and not marker_visibility.get(
                name, True
            ):
                continue
            times = np.asarray(dataset[var_name].values, dtype=float)
            self._marker_series[name] = (times, QColor(colour))

        self.update()

    def clear_selection(self):
        """Drop the shaded range — for when new data replaces the old."""
        self._selection = None
        self.update()

    def values_at(self, time_s: float) -> list[tuple[str, float]]:
        """Each visible series' value at the bin containing *time_s*."""
        out = []
        for x_axis, y_values, _, label in self._series:
            samples = np.asarray(x_axis, dtype=float)
            if samples.size == 0:
                continue
            index = int(
                np.clip(np.searchsorted(samples, time_s), 0, samples.size - 1)
            )
            out.append((label, float(np.asarray(y_values)[index])))
        return out

    def _plot_rect(self):
        left, top, right, bottom = _PLOT_MARGINS
        return self.rect().adjusted(left, top, -right, -bottom)

    def _time_from_x(self, x_pos) -> float | None:
        if not self._series or self._x_min is None or self._x_max is None:
            return None
        rect = self._plot_rect()
        if self._x_max <= self._x_min:
            return self._x_min
        if x_pos <= rect.left():
            return self._x_min
        if x_pos >= rect.right():
            return self._x_max
        frac = (x_pos - rect.left()) / rect.width()
        return self._x_min + frac * (self._x_max - self._x_min)

    def _x_from_time(self, value: float, rect) -> int:
        span = self._x_max - self._x_min
        span = span if span else 1.0
        return int(rect.left() + (value - self._x_min) * rect.width() / span)

    @staticmethod
    def _event_pos(event):
        """Mouse position, on both Qt bindings.

        ``event.pos()`` exists in Qt5 and (deprecated but present) in Qt6,
        unlike ``position()`` which is Qt6-only.
        """
        return event.pos()

    def _update_selector_from_event(self, event):
        pos = self._event_pos(event)
        if not self._plot_rect().contains(pos):
            return
        self._selector_time = self._time_from_x(pos.x())
        if self._selector_time is not None:
            self.selector_changed.emit(float(self._selector_time))
            if self._dragging:
                self._drag_end_time = float(self._selector_time)
        self.update()

    def mouseMoveEvent(self, event):
        self._update_selector_from_event(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = self._event_pos(event)
        if not self._plot_rect().contains(pos):
            return
        self._dragging = True
        if self._selection is not None:
            self._selection = None
            self.selection_cleared.emit()
        self._drag_start_time = self._time_from_x(pos.x())
        self._drag_end_time = self._drag_start_time
        self._selector_time = self._drag_start_time
        if self._selector_time is not None:
            self.selector_changed.emit(float(self._selector_time))
        self.update()

    def mouseReleaseEvent(self, event):
        if not self._dragging:
            return
        pos = self._event_pos(event)
        if self._plot_rect().contains(pos):
            self._drag_end_time = self._time_from_x(pos.x())

        if (
            self._drag_start_time is not None
            and self._drag_end_time is not None
        ):
            lo = min(float(self._drag_start_time), float(self._drag_end_time))
            hi = max(float(self._drag_start_time), float(self._drag_end_time))
            if hi > lo:
                self._selection = (lo, hi)
                self.selection_range_changed.emit(lo, hi)

        self._dragging = False
        self._drag_start_time = None
        self._drag_end_time = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(MPL.FIG_BG))

        rect = self._plot_rect()
        painter.fillRect(rect, QColor(MPL.AXES_BG))
        painter.setPen(QPen(QColor(MPL.SPINE), 1))
        painter.drawRect(rect)

        if not self._series or rect.width() <= 0 or rect.height() <= 0:
            painter.setPen(QPen(QColor(C.TEXT_DIM)))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No trace — set the range and click Apply",
            )
            painter.end()
            return

        values = np.concatenate(
            [np.asarray(s[1], dtype=float).ravel() for s in self._series]
        )
        v_min, v_max = float(np.nanmin(values)), float(np.nanmax(values))
        if v_min == v_max:
            v_min, v_max = v_min - 1.0, v_max + 1.0
        v_span = v_max - v_min or 1.0

        def _y(value: float) -> int:
            return int(
                rect.bottom() - (value - v_min) * rect.height() / v_span
            )

        painter.setClipRect(rect)

        if self._selection is not None:
            lo, hi = self._selection
            x_lo = self._x_from_time(max(lo, self._x_min), rect)
            x_hi = self._x_from_time(min(hi, self._x_max), rect)
            if x_hi > x_lo:
                shade = QColor(MPL.SELECTION)
                shade.setAlpha(MPL.SELECTION_FILL_ALPHA)
                painter.fillRect(
                    x_lo, rect.top(), x_hi - x_lo, rect.height(), shade
                )
                edge = QColor(MPL.SELECTION)
                edge.setAlpha(MPL.SELECTION_EDGE_ALPHA)
                painter.setPen(QPen(edge, 1))
                painter.drawLine(x_lo, rect.top(), x_lo, rect.bottom())
                painter.drawLine(x_hi, rect.top(), x_hi, rect.bottom())

        painter.setClipping(False)

        exponent = 0
        if v_max >= 1000:
            exponent = int(np.floor(np.log10(abs(v_max))) // 3 * 3)
        scale = 10.0**exponent
        x_span = (self._x_max - self._x_min) or 1.0
        x_decimals = int(np.clip(np.ceil(-np.log10(x_span / 4)) + 1, 0, 6))

        painter.setPen(QPen(QColor(MPL.TICK), 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        painter.drawLine(rect.bottomLeft(), rect.topLeft())
        for tick in np.linspace(self._x_min, self._x_max, 5):
            x = self._x_from_time(float(tick), rect)
            painter.drawLine(x, rect.bottom(), x, rect.bottom() + 5)
            painter.drawText(
                x - 18, rect.bottom() + 18, f"{tick:.{x_decimals}f}"
            )
        for tick in np.linspace(v_min, v_max, 5):
            y = _y(float(tick))
            painter.drawLine(rect.left() - 5, y, rect.left(), y)
            painter.drawText(
                rect.left() - 48, y + 4, f"{tick / scale:.3g}".rjust(7)
            )
        if exponent:
            painter.drawText(
                rect.left() - 48, rect.top() - 8, f"×10^{exponent}"
            )

        painter.setClipRect(rect)

        # Series
        for _index, (x_axis, y_values, colour, _label) in enumerate(
            self._series
        ):
            samples = np.asarray(x_axis, dtype=float)
            y_data = np.asarray(y_values, dtype=float)
            if samples.size != y_data.size:
                continue
            points = [
                (self._x_from_time(float(t), rect), _y(float(v)))
                for t, v in zip(samples, y_data, strict=True)
            ]
            painter.setPen(QPen(colour, 1))
            for start, end in zip(points, points[1:], strict=False):
                painter.drawLine(*start, *end)

        # Marker overlays
        for times, colour in self._marker_series.values():
            if len(times) == 0:
                continue
            painter.setPen(QPen(colour, 1, Qt.PenStyle.DashLine))
            y_zero = _y(0.0)
            for marker_time in times:
                if not (self._x_min <= marker_time <= self._x_max):
                    continue
                x = self._x_from_time(float(marker_time), rect)
                painter.drawLine(x, rect.top(), x, y_zero)

        for index, (_, _, colour, label) in enumerate(self._series):
            text_x, text_y = rect.left() + 10, rect.top() + 12 + index * 16
            box = painter.fontMetrics().boundingRect(label)
            box.moveTo(text_x - 3, text_y - box.height() + 2)
            box.adjust(0, -1, 6, 3)
            plate = QColor(MPL.LEGEND_BG)
            plate.setAlpha(min(255, MPL.LEGEND_BG_ALPHA))
            painter.fillRect(box, plate)
            painter.setPen(QPen(colour))
            painter.drawText(text_x, text_y, label)

        # Selector
        if (
            self._selector_time is not None
            and self._x_min <= self._selector_time <= self._x_max
        ):
            painter.setPen(QPen(QColor(C.TEXT), 1.5, Qt.PenStyle.DashLine))
            x = self._x_from_time(float(self._selector_time), rect)
            painter.drawLine(x, rect.top(), x, rect.bottom())

        # Axis titles live in the margins.
        painter.setClipping(False)
        painter.setPen(QPen(QColor(MPL.TICK), 1))
        painter.drawText(rect.center().x() - 30, self.height() - 7, "Time (s)")
        painter.save()
        painter.translate(14, rect.center().y())
        painter.rotate(-90)
        painter.drawText(0, -5, "Photon count")
        painter.restore()
        painter.end()
