import traceback

from qtpy.QtCore import Qt, QThreadPool, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_flopa.core.processing.trace import reconstruct_trace
from napari_flopa.ui.state import FlopaState
from napari_flopa.ui.style import S, apply_style
from napari_flopa.ui.utils.threading import Worker
from napari_flopa.ui.widgets.line_plot import LinePlotWidget
from napari_flopa.ui.widgets.status_label import StatusLabel

_MARKERS = (
    ("line_start", "Line start"),
    ("line_stop", "Line stop"),
    ("frame_start", "Frame start"),
)

# Bin-width step (ms) per window length (s) (stop-start time min difference, step)
_BIN_STEPS = (
    (10.0, 10.0),
    (5.0, 5.0),
    (1.0, 1.0),
    (0.5, 0.1),
    (0.0, 0.01),
)


def bin_step_for_window(window_s: float) -> float:
    """Sensible bin-width step, in ms, for a window of *window_s* seconds."""
    for threshold, step in _BIN_STEPS:
        if window_s >= threshold:
            return step
    return _BIN_STEPS[-1][1]


class TracePanel(QWidget):
    """Trace tab — intensity vs time, with marker overlays."""

    def __init__(self, state: FlopaState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer
        self.threadpool = QThreadPool()

        self._result = None  # last trace dataset
        self._plotted_window = (0.0, 10.0)
        self._channel_checks: dict[int, QCheckBox] = {}
        self._running = False
        self._resting_status = "Load a PTU file in the File tab."

        self._build_ui()
        self.state.file_loaded.connect(self._on_file_loaded)
        self._on_file_loaded()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        params = QGroupBox("Trace")
        apply_style(params, S.GROUP_PRIMARY)
        form = QFormLayout(params)
        form.setContentsMargins(6, 2, 6, 4)
        form.setSpacing(4)

        self._file_lbl = QLabel()
        self._file_lbl.setStyleSheet(S.HINT)
        self._file_lbl.setWordWrap(True)
        form.addRow("PTU file:", self._file_lbl)

        self._start_spin = self._time_spin(0.0)
        self._stop_spin = self._time_spin(10.0)
        self._det_spin = QSpinBox()
        self._det_spin.setRange(1, 64)
        self._det_spin.setValue(1)
        self._bin_spin = QDoubleSpinBox()
        self._bin_spin.setRange(0.001, 10_000.0)
        self._bin_spin.setDecimals(3)
        self._bin_spin.setSingleStep(0.01)
        self._bin_spin.setValue(1.0)
        self._bin_spin.setToolTip("Width of one time bin, in milliseconds")

        for label, widget in (
            ("Start time (s):", self._start_spin),
            ("Stop time (s):", self._stop_spin),
            ("Max detector:", self._det_spin),
            ("Bin width (ms):", self._bin_spin),
        ):
            form.addRow(label, widget)
            widget.valueChanged.connect(self._mark_stale)
        # The usable bin width depends on how long the window is.
        self._start_spin.valueChanged.connect(self._tune_bin_spin)
        self._stop_spin.valueChanged.connect(self._tune_bin_spin)
        self._tune_bin_spin()
        root.addWidget(params)

        run_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip(
            "Reconstruct the trace for this time window"
        )
        self._apply_btn.clicked.connect(self._on_apply)
        run_row.addWidget(self._apply_btn)

        self._stale = QLabel("●")
        self._stale.setFixedWidth(14)
        self._stale.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stale.setStyleSheet(S.STALE_INACTIVE)
        self._stale.setVisible(False)
        run_row.addWidget(self._stale)
        run_row.addStretch()
        root.addLayout(run_row)

        display = QHBoxLayout()
        display.setSpacing(6)
        self._sum_check = QCheckBox("Sum channels")
        self._sum_check.toggled.connect(self._redraw)
        display.addWidget(self._sum_check)
        self._channel_row = QHBoxLayout()
        self._channel_row.setSpacing(4)
        display.addLayout(self._channel_row)
        display.addStretch()
        root.addLayout(display)

        marker_row = QHBoxLayout()
        marker_row.setSpacing(6)
        self._marker_checks: dict[str, QCheckBox] = {}
        for name, label in _MARKERS:
            chk = QCheckBox(f"{label} markers")
            chk.setChecked(name == "frame_start")
            chk.toggled.connect(self._redraw)
            self._marker_checks[name] = chk
            marker_row.addWidget(chk)
        marker_row.addStretch()
        root.addLayout(marker_row)

        self._plot = LinePlotWidget()
        self._plot.selector_changed.connect(self._on_selector_moved)
        self._plot.selection_range_changed.connect(self._on_range_selected)
        self._plot.selection_cleared.connect(self._on_selection_cleared)
        root.addWidget(self._plot, stretch=1)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Selector (s):"))
        self._selector_spin = QDoubleSpinBox()
        self._selector_spin.setRange(-1e9, 1e9)
        self._selector_spin.setDecimals(4)
        self._selector_spin.setReadOnly(True)
        self._selector_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        sel_row.addWidget(self._selector_spin)
        self._selector_values = QLabel()
        self._selector_values.setStyleSheet(S.HINT)
        self._selector_values.setToolTip(
            "Photon count of each visible detector in the bin under the cursor"
        )
        sel_row.addWidget(self._selector_values)
        sel_row.addStretch()
        root.addLayout(sel_row)

        self._status = StatusLabel("Load a PTU file in the File tab.")
        root.addWidget(self._status)

    def _tune_bin_spin(self, *_):
        """Match the bin-width step and ceiling to the current time window.

        A bin can never be longer than the window, and the step that feels
        right for a 60 s trace is useless for a 200 ms one.
        """
        window_s = max(self._stop_spin.value() - self._start_spin.value(), 0.0)
        window_ms = window_s * 1_000.0
        self._bin_spin.setSingleStep(bin_step_for_window(window_s))
        if window_ms > 0:
            self._bin_spin.setMaximum(window_ms)
        self._bin_spin.setToolTip(
            f"Width of one time bin, in milliseconds "
            f"(window {window_s:g} s → step "
            f"{bin_step_for_window(window_s):g} ms)"
        )

    @staticmethod
    def _time_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 999.999)
        spin.setDecimals(3)
        spin.setSingleStep(0.01)
        spin.setValue(value)
        return spin

    def _ptu_path(self):
        return self.state.file_runtime().get("ptu_path")

    @Slot()
    def _on_file_loaded(self):
        """Follow the File tab's selection; a new file invalidates the trace."""
        path = self._ptu_path()
        self._file_lbl.setText(path.name if path else "No file loaded")
        self._apply_btn.setEnabled(path is not None and not self._running)
        if path is not None:
            self._resting_status = "Set the time window and click Apply."
            self._status.info(self._resting_status)
            self._mark_stale()

    def _mark_stale(self, *_):
        """Red dot: the plot no longer matches the parameters."""
        if self._result is None:
            return
        self._stale.setStyleSheet(S.STALE_STALE)
        self._stale.setToolTip("Parameters changed — click Apply to refresh.")
        self._stale.setVisible(True)

    def _mark_fresh(self):
        self._stale.setStyleSheet(S.STALE_FRESH)
        self._stale.setToolTip("Trace matches the current parameters.")
        self._stale.setVisible(True)

    def _on_apply(self):
        path = self._ptu_path()
        if path is None:
            self._status.warn("No PTU file loaded in the File tab.")
            return
        if self._stop_spin.value() <= self._start_spin.value():
            self._status.warn("Stop time must be greater than start time.")
            return

        runtime = self.state.file_runtime()
        params = dict(
            start_time=self._start_spin.value(),
            stop_time=self._stop_spin.value(),
            max_detector=self._det_spin.value(),
            bin_width_s=self._bin_spin.value() / 1_000.0,
            chunk_size=runtime.get("chunk_size") or None,
        )
        params = {k: v for k, v in params.items() if v is not None}

        self._running = True
        self._apply_btn.setEnabled(False)
        self._status.info(f"Reading {path.name}…")

        worker = Worker(reconstruct_trace, path, **params)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self._on_error)
        worker.signals.finished.connect(self._on_finished)
        self.threadpool.start(worker)

    @Slot(object)
    def _on_result(self, dataset):
        self._result = dataset
        self._plotted_window = (
            self._start_spin.value(),
            self._stop_spin.value(),
        )
        self._plot.clear_selection()
        self._rebuild_channel_checks(dataset)
        self._redraw()
        self._mark_fresh()
        n_bins = dataset.sizes.get("time", 0)
        n_det = dataset.sizes.get("channel", 0)
        self._resting_status = (
            f"{n_bins:,} bins × {n_det} detector(s) "
            f"over {self._start_spin.value():g}–{self._stop_spin.value():g} s."
        )
        self._status.info(self._resting_status)

    @Slot(tuple)
    def _on_error(self, error_tuple):
        _, value, tb = error_tuple
        self._status.error(f"Trace failed: {value}")
        traceback.print_exception(type(value), value, None)
        print(tb)

    @Slot()
    def _on_finished(self):
        self._running = False
        self._apply_btn.setEnabled(self._ptu_path() is not None)

    def _rebuild_channel_checks(self, dataset):
        """One checkbox per detector present in the result."""
        for chk in self._channel_checks.values():
            self._channel_row.removeWidget(chk)
            chk.deleteLater()
        self._channel_checks.clear()
        if dataset is None or "photon_count" not in dataset.data_vars:
            return
        for channel_id in dataset["photon_count"].coords["channel"].values:
            chk = QCheckBox(f"D{int(channel_id)}")
            chk.setChecked(True)
            chk.toggled.connect(self._redraw)
            self._channel_checks[int(channel_id)] = chk
            self._channel_row.addWidget(chk)

    def _redraw(self, *_):
        """Repaint from the cached result — never re-reads the file."""
        if self._result is None:
            return
        start, stop = self._plotted_window
        self._plot.set_values(
            self._result,
            start_time=start,
            stop_time=stop,
            visible_channels=[
                cid
                for cid, chk in self._channel_checks.items()
                if chk.isChecked()
            ],
            sum_selected=self._sum_check.isChecked(),
            marker_visibility={
                name: chk.isChecked()
                for name, chk in self._marker_checks.items()
            },
        )

    @Slot(float)
    def _on_selector_moved(self, value: float):
        self._selector_spin.setValue(float(value))
        readout = "   ".join(
            f"{label.replace('Detector ', 'D')}: {count:,.0f}"
            for label, count in self._plot.values_at(float(value))
        )
        self._selector_values.setText(readout)

    @Slot(float, float)
    def _on_range_selected(self, start: float, stop: float):
        """A drag on the plot becomes the next time window."""
        self._start_spin.setValue(float(start))
        self._stop_spin.setValue(float(stop))
        self._status.info("Range selected — click Apply to reconstruct it.")

    @Slot()
    def _on_selection_cleared(self):
        """Clicking without dragging drops the range — and its message."""
        self._status.info(self._resting_status)
