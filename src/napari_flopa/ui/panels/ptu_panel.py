import traceback
from pathlib import Path

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from tttrkit.ptuio.reconstructor import ScanConfig
from tttrkit.ptuio.utils import estimate_bidirectional_shift
from qtpy.QtCore import Qt, QThreadPool, Signal, Slot
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_flopa.core import provenance
from napari_flopa.core.demo import load_demo
from napari_flopa.core.io.config import (
    build_scan_config_dict,
    load_config,
    save_config,
)
from napari_flopa.core.io.loader import (
    format_ptu_header,
    read_ptu_file,
)
from napari_flopa.core.io.markers import (
    _format_marker_suggestions,
    analyze_marker_distribution,
    get_markers,
)
from napari_flopa.core.logger import ProgressLogger
from napari_flopa.core.processing.reconstruction import (
    DEFAULT_CHUNK_SIZE,
    reconstruct_ptu_to_dataset,
)
from napari_flopa.ui.state import FlopaState
from napari_flopa.ui.style import MPL, S, apply_style
from napari_flopa.ui.utils.threading import Worker


def _fmt_factor(c: complex) -> str:
    """Config-file form of a calibration factor, e.g. ``0.8644-0.0542j``."""
    return str(complex(c)).strip("()")


class PtuPanel(QWidget):
    """
    File tab: Load PTU file, inspect header, configure scan parameters,
    run reconstruction.
    """

    reconstruction_finished = Signal(object)  # xr.Dataset
    # (done_chunks, total_chunks) — emitted from the worker thread, delivered
    # to the main thread so the progress bar updates safely.
    reconstruction_progress = Signal(int, int)

    def __init__(self, state: FlopaState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer
        self.threadpool = QThreadPool()

        self.ptu_data = None
        self.ptu_filepath = None
        self.shift_plot_data = None
        self._log_buffer: list[str] = []

        # Parameter provenance: name -> 'metadata'|'default'|'user', plus the
        # coloured dot widgets. `_autofill` guards programmatic setValue() calls
        # so auto-filling from the header doesn't mark fields as user-edited.
        self._param_source: dict[str, str] = {}
        self._param_dots: dict[str, object] = {}
        self._autofill = False

        # Calibration factor carried by the loaded config (None = not in the
        # file); pushed to the shared state when reconstruction finishes, so
        # the Phasor tab picks it up.
        self._cfg_calib_factor: complex | None = None

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Parameter provenance                                                 #
    # ------------------------------------------------------------------ #

    def _set_param_source(self, name: str, source: str):
        """Record a parameter's source and recolour its dot."""
        self._param_source[name] = source
        dot = self._param_dots.get(name)
        if dot is not None:
            dot.setStyleSheet(S.PROV_DOT.get(source, ""))
            dot.setToolTip(f"Value source: {source}")

    def _on_param_edited(self, name: str):
        """A manual spinbox edit flips the field to 'user' (ignored while auto-filling)."""
        if not self._autofill:
            self._set_param_source(name, provenance.USER)

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # --- File loading ---
        file_group = QGroupBox("Load Data")
        apply_style(file_group, S.GROUP_PRIMARY)
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel("No file selected.")
        # Long .ptu names would otherwise stretch the whole panel wide; wrap
        # (breaking anywhere, since filenames have no spaces) so the label's
        # width stops driving the panel's minimum width.
        self.file_label.setWordWrap(True)
        self.file_label.setMinimumWidth(1)
        self.file_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        btn_row = QHBoxLayout()
        self.select_ptu_btn = QPushButton("Read PTU...")
        self.select_ptu_btn.clicked.connect(self._select_ptu_file)
        self.load_demo_btn = QPushButton("Load Demo")
        self.load_demo_btn.setToolTip(
            "Load the bundled demo PTU file with preset scan parameters"
        )
        self.load_demo_btn.clicked.connect(self._on_load_demo)
        btn_row.addWidget(self.select_ptu_btn)
        btn_row.addWidget(self.load_demo_btn)
        btn_row.addStretch()
        file_layout.addWidget(self.file_label)
        file_layout.addLayout(btn_row)
        main_layout.addWidget(file_group)

        # --- Header info (summary only, full header via button) ---
        self.header_group = QGroupBox("Header Info")
        apply_style(self.header_group, S.GROUP_PRIMARY)
        header_layout = QVBoxLayout(self.header_group)
        self.header_info = QTextEdit()
        self.header_info.setReadOnly(True)
        self.header_info.setFixedHeight(100)
        apply_style(self.header_info, S.LOG)  # match the log editor look
        header_layout.addWidget(self.header_info)

        marker_row = QHBoxLayout()
        self.markers_btn = QPushButton("Analyze Markers")
        self.markers_btn.setToolTip(
            "Read marker events to suggest scan dimensions"
        )
        self.markers_btn.clicked.connect(self._show_markers)
        self.full_header_btn = QPushButton("Full Header...")
        self.full_header_btn.setToolTip("Show complete raw header tags")
        self.full_header_btn.clicked.connect(self._show_full_header)
        marker_row.addWidget(self.markers_btn)
        marker_row.addWidget(self.full_header_btn)
        marker_row.addStretch()
        header_layout.addLayout(marker_row)
        main_layout.addWidget(self.header_group)
        self.header_group.setVisible(False)

        # --- Scan configuration ---
        self.config_group = self._build_config_group()
        main_layout.addWidget(self.config_group)
        self.config_group.setVisible(False)

        # --- Reconstruction ---
        self.recon_group = QGroupBox("Reconstruction")
        apply_style(self.recon_group, S.GROUP_PRIMARY)
        recon_layout = QVBoxLayout(self.recon_group)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output:"))
        self.out_combo = QComboBox()
        self.out_combo.addItems(
            [
                "Intensity (I)",
                "I + τ",
                "All",
            ]
        )
        self.out_combo.setCurrentIndex(2)
        self.out_combo.setToolTip(
            "Reconstruction outputs:\n"
            "Intensity (I): photon-count image only\n"
            "I + τ: intensity + mean arrival-time (lifetime)\n"
            "All: also compute phasor coordinates + TCSPC decay"
        )
        # Without this the combo's minimum fits its longest item
        # ("All (+ Phasor & Decay)"), which alone forced the whole File panel
        # to ~750px min width. AdjustToMinimumContentsLength + a small minimum
        # lets it shrink (eliding) when the dock is narrow; stretch=1 makes it
        # expand to show the full text when there's room.
        # self.out_combo.setSizeAdjustPolicy(
        #     QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLength
        # )
        # self.out_combo.setMinimumContentsLength(8)
        # output_row.addWidget(self.out_combo, 1)
        output_row.addWidget(self.out_combo, 1)

        output_row.addWidget(QLabel("Chunk size:"))
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(100_000, 100_000_000)
        self.chunk_size_spin.setSingleStep(100_000)
        self.chunk_size_spin.setGroupSeparatorShown(True)
        self.chunk_size_spin.setValue(DEFAULT_CHUNK_SIZE)
        self.chunk_size_spin.setToolTip(
            "TTTR records read per iteration "
            f"(default {DEFAULT_CHUNK_SIZE:,}).\n"
            "Smaller = less memory and finer progress steps; "
            "larger = fewer iterations.\n"
            "Does not change the reconstructed data."
        )
        output_row.addWidget(self.chunk_size_spin, 1)

        self.reconstruct_btn = QPushButton("Reconstruct")
        self.reconstruct_btn.clicked.connect(self._run_reconstruction)
        output_row.addWidget(self.reconstruct_btn)
        recon_layout.addLayout(output_row)

        self.recon_progress = QProgressBar()
        self.recon_progress.setTextVisible(True)
        self.recon_progress.setVisible(False)
        recon_layout.addWidget(self.recon_progress)
        self.reconstruction_progress.connect(self._on_reconstruction_progress)

        main_layout.addWidget(self.recon_group)
        self.recon_group.setVisible(False)

        # --- Info / log ---
        self.info_group = QGroupBox("Log")
        apply_style(self.info_group, S.GROUP_PRIMARY)
        info_layout = QVBoxLayout(self.info_group)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        apply_style(self.log_text, S.LOG)
        info_layout.addWidget(self.log_text)
        # Stretch factor 1 → the log fills the leftover panel height (growing
        # from its 120px minimum). While it's still hidden on init, the trailing
        # addStretch() below absorbs the space instead, so nothing balloons.
        main_layout.addWidget(self.info_group, 1)
        self.info_group.setVisible(False)

        main_layout.addStretch()

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("Scan Configuration")
        apply_style(group, S.GROUP_PRIMARY)
        main_layout = QVBoxLayout(group)

        # Single-column form. napari's spinboxes have a large minimum width, so
        # a 2-column grid forced the whole panel wide; stacking every scalar
        # parameter in one column keeps the dock narrow and the labels aligned.
        form = QFormLayout()
        form.setVerticalSpacing(4)

        def _spin_row(name, spinbox):
            # Row = [spinbox] [provenance dot] [stretch]; registers the field
            # so its source dot updates and flips to "user" on manual edits.
            spinbox.setMinimumWidth(40)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(spinbox)
            dot = QLabel("●")
            self._param_dots[name] = dot
            self._set_param_source(name, provenance.DEFAULT)
            spinbox.valueChanged.connect(
                lambda *_: self._on_param_edited(name)
            )
            row.addWidget(dot)
            row.addStretch()
            return row

        self.lines_spin = QSpinBox()
        self.lines_spin.setRange(1, 10000)
        self.lines_spin.setValue(512)
        form.addRow("Lines:", _spin_row("lines", self.lines_spin))

        self.pixels_spin = QSpinBox()
        self.pixels_spin.setRange(1, 10000)
        self.pixels_spin.setValue(512)
        form.addRow("Pixels:", _spin_row("pixels", self.pixels_spin))

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 1000)
        self.frames_spin.setValue(1)
        form.addRow("Frames:", _spin_row("frames", self.frames_spin))

        self.tcspc_bins_spin = QSpinBox()
        self.tcspc_bins_spin.setRange(1, 65536)
        self.tcspc_bins_spin.setValue(4096)
        form.addRow(
            "TCSPC Bins:", _spin_row("tcspc_bins", self.tcspc_bins_spin)
        )

        self.max_detector_spin = QSpinBox()
        self.max_detector_spin.setRange(1, 128)
        self.max_detector_spin.setValue(2)
        form.addRow(
            "Max detectors:", _spin_row("max_detector", self.max_detector_spin)
        )

        # Rep. rate / N Sequences / Accumulations continue in the same column,
        # directly under Max Det.
        self.frep_spin = QDoubleSpinBox()
        self.frep_spin.setRange(0.001, 500.0)
        self.frep_spin.setValue(40.0)
        self.frep_spin.setDecimals(3)
        self.frep_spin.setToolTip(
            "Laser/excitation repetition rate — auto-filled from file header.\n"
            "Used for phasor calibration and lifetime calculation."
        )
        self.frep_spin.valueChanged.connect(lambda v: self.state.set_frep(v))
        form.addRow("Rep. rate (MHz):", _spin_row("frep", self.frep_spin))

        self.sequences_spin = QSpinBox()
        self.sequences_spin.setRange(1, 16)
        self.sequences_spin.setValue(1)
        self.sequences_spin.valueChanged.connect(
            self._update_accumulation_widgets
        )
        form.addRow(
            "N sequences:", _spin_row("sequences", self.sequences_spin)
        )

        main_layout.addLayout(form)

        # Accumulations — its own row (label + scroll in an HBox) rather than a
        # form field, so it spans the full panel width instead of the narrow
        # form-field column.
        self.accu_scroll = QScrollArea()
        self.accu_scroll.setWidgetResizable(True)
        # self.accu_scroll.setMinimumHeight(60)
        # self.accu_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.accu_container = QWidget()
        self.accu_container_layout = QVBoxLayout(self.accu_container)
        self.accu_container_layout.setContentsMargins(2, 2, 2, 2)
        self.accu_container_layout.setSpacing(2)
        self.accu_scroll.setWidget(self.accu_container)
        self.accu_spinboxes = []
        accu_label = QLabel("Accumulation:")
        accu_label.setToolTip("Number of line accumulations per sequence")

        accu_row = QHBoxLayout()
        accu_row.addWidget(accu_label, alignment=Qt.AlignTop)
        accu_row.addWidget(self.accu_scroll)
        main_layout.addLayout(accu_row)

        # --- Bidirectional (nested, checkable subsection) ---
        # GROUP_NESTED sets no background-color, so the box is transparent and
        # inherits the Scan Configuration tint — its content area matches the
        # surrounding box exactly.
        self.bidir_group = QGroupBox("Bidirectional Scan")
        apply_style(self.bidir_group, S.GROUP_CHECKABLE)
        self.bidir_group.setCheckable(True)
        self.bidir_group.setChecked(False)
        bidir_layout = QHBoxLayout(self.bidir_group)
        bidir_layout.setContentsMargins(0, 0, 0, 0)

        bidir_layout.addWidget(QLabel("Phase Shift:"))
        self.bidir_phase_spin = QDoubleSpinBox()
        self.bidir_phase_spin.setRange(-0.2, 0.2)
        self.bidir_phase_spin.setSingleStep(0.0001)
        self.bidir_phase_spin.setDecimals(5)
        self.bidir_phase_spin.setValue(0.0)
        # self.bidir_phase_spin.setMinimumWidth(70)
        bidir_layout.addWidget(self.bidir_phase_spin)
        # bidir_layout.addStretch()

        # Compact Est./Plot buttons, matched to the same small height.
        self.estimate_btn = QPushButton("Est.")
        self.estimate_btn.setToolTip("Estimate bidirectional phase shift")
        self.estimate_btn.setFixedHeight(22)
        self.estimate_btn.setStyleSheet(S.BTN_SMALL)
        self.estimate_btn.clicked.connect(self._on_estimate_shift)
        bidir_layout.addWidget(self.estimate_btn)

        self.plot_shift_btn = QPushButton("Plot")
        self.plot_shift_btn.setEnabled(False)
        self.plot_shift_btn.setToolTip("Plot shift correlation curve")
        self.plot_shift_btn.setFixedHeight(22)
        self.plot_shift_btn.setStyleSheet(S.BTN_SMALL)
        self.plot_shift_btn.clicked.connect(self._on_plot_shift)
        bidir_layout.addWidget(self.plot_shift_btn)
        bidir_layout.addStretch()

        main_layout.addWidget(self.bidir_group)

        # --- Load / Save scan config (full-width row, at the end) ---
        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.setToolTip(
            "Load scan parameters from a JSON file"
        )
        self.load_config_btn.clicked.connect(self._on_load_config_json)
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.setToolTip("Save scan parameters to a JSON file")
        self.save_config_btn.clicked.connect(self._on_save_config_json)
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(self.load_config_btn, 1)
        cfg_row.addWidget(self.save_config_btn, 1)
        main_layout.addLayout(cfg_row)

        self._update_accumulation_widgets()
        return group

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _log_start(self):
        """Begin accumulating a new log block."""
        self._log_buffer = []

    def _log_line(self, text: str):
        """Add a line to the current log block buffer."""
        self._log_buffer.append(text)

    def _log_commit(self):
        """Prepend the buffered block as one unit to the Info box."""
        if not self._log_buffer:
            return
        block = "\n".join(self._log_buffer)
        current = self.log_text.toPlainText()
        new_content = (
            block if not current else f"{block}\n{'-' * 40}\n{current}"
        )
        self.log_text.setPlainText(new_content)
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        self.log_text.setTextCursor(cursor)
        self._log_buffer = []

    # keep a single-call shortcut for one-liner messages
    def _log(self, text: str):
        self._log_start()
        self._log_line(text)
        self._log_commit()

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _select_ptu_file(self):
        filepath_str, _ = QFileDialog.getOpenFileName(
            self, "Select PTU File", "", "PicoQuant Files (*.ptu)"
        )  # type: ignore
        if not filepath_str:
            return
        self.ptu_filepath = Path(filepath_str)
        self.file_label.setText(f"file: {self.ptu_filepath.name}")
        try:
            self.ptu_data = read_ptu_file(str(self.ptu_filepath), header=False)
            tags = self.ptu_data["header"]
            constants = self.ptu_data["constants"]
            csrc = self.ptu_data.get("constants_source", {})

            # Summary (no full header) — full header via button
            summary = format_ptu_header(
                tags, constants, full_header=False, constants_source=csrc
            )
            self.header_info.setPlainText(summary)

            # Auto-fill from header (guarded so setValue doesn't mark 'user')
            self._autofill = True
            px_x = tags.get("ImgHdr_PixX")
            px_y = tags.get("ImgHdr_PixY")
            n_frames = tags.get("ImgHdr_NumberOfFrames")
            if isinstance(px_x, (int, float)):
                self.pixels_spin.setValue(int(px_x))
                self._set_param_source("pixels", provenance.METADATA)
            if isinstance(px_y, (int, float)):
                self.lines_spin.setValue(int(px_y))
                self._set_param_source("lines", provenance.METADATA)
            if isinstance(n_frames, (int, float)) and n_frames > 0:
                self.frames_spin.setValue(int(n_frames))
                self._set_param_source("frames", provenance.METADATA)
            self.tcspc_bins_spin.setValue(constants.get("tcspc_bins", 4096))
            self._set_param_source(
                "tcspc_bins", csrc.get("tcspc_bins", provenance.DEFAULT)
            )
            rep = constants.get("repetition_rate") or constants.get(
                "sync_rate_hz"
            )
            if rep:
                self.frep_spin.setValue(float(rep) / 1e6)
                self._set_param_source(
                    "frep", csrc.get("repetition_rate", provenance.DEFAULT)
                )
            self._autofill = False

            self.header_group.setVisible(True)
            self.config_group.setVisible(True)
            self.recon_group.setVisible(True)
            self.info_group.setVisible(True)
            self.log_text.clear()
            self.shift_plot_data = None
            self.plot_shift_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to read PTU header:\n{e}"
            )

    def _on_load_demo(self):
        try:
            params, ptu_path = load_demo()
        except Exception as e:
            QMessageBox.critical(self, "Demo Error", str(e))
            return

        try:
            self.ptu_filepath = ptu_path
            self.file_label.setText(f"PTU: {ptu_path.name}")
            self.ptu_data = read_ptu_file(str(ptu_path), header=False)
            tags = self.ptu_data["header"]
            constants = self.ptu_data["constants"]
            csrc = self.ptu_data.get("constants_source", {})
            summary = format_ptu_header(
                tags, constants, full_header=False, constants_source=csrc
            )
            self.header_info.setPlainText(summary)

            # Apply the demo scan params (marked 'user'), via the same path as
            # Load Config. The params' ptu_filename is ignored here — the file
            # was already resolved/loaded above by load_demo().
            self._apply_config(params, source=provenance.USER)

            # File-derived values override the config for tcspc bins + rep. rate.
            self._autofill = True
            self.tcspc_bins_spin.setValue(constants.get("tcspc_bins", 4096))
            self._set_param_source(
                "tcspc_bins", csrc.get("tcspc_bins", provenance.DEFAULT)
            )
            rep = constants.get("repetition_rate") or constants.get(
                "sync_rate_hz"
            )
            if rep:
                self.frep_spin.setValue(float(rep) / 1e6)
                self._set_param_source(
                    "frep", csrc.get("repetition_rate", provenance.DEFAULT)
                )
            self._autofill = False

            self.header_group.setVisible(True)
            self.config_group.setVisible(True)
            self.recon_group.setVisible(True)
            self.info_group.setVisible(True)
            self.log_text.clear()
            self.shift_plot_data = None
            self.plot_shift_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(
                self, "Demo Load Error", f"Failed to load demo PTU:\n{e}"
            )

    def _show_full_header(self):
        if not self.ptu_data:
            return
        tags = self.ptu_data["header"]
        constants = self.ptu_data["constants"]
        full_text = format_ptu_header(tags, constants, full_header=True)
        dlg = QDialog(self)
        dlg.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        dlg.setWindowTitle("Full PTU Header")
        dlg.resize(600, 500)
        layout = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        apply_style(te, S.LOG)  # unify colours with the log editor
        te.setPlainText(full_text)
        layout.addWidget(te)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()

    def _show_markers(self):
        if not self.ptu_data:
            self._log("Analyze Markers: no file loaded.")
            return
        self._log_start()
        self._log_line("Analyzing markers...")
        QApplication.processEvents()
        try:
            dist = get_markers(self.ptu_data["reader"], chunk_limit=20)
            if "error" in dist:
                self._log_line(dist["error"])
                self._log_commit()
                return
            analysis = analyze_marker_distribution(dist)
            text = _format_marker_suggestions(analysis)
            self._log_line(text)
            self._log_commit()

            # Auto-apply if only one suggestion
            suggestions = analysis.get("suggestions", [])
            if len(suggestions) == 1:
                lines, accum = suggestions[0]
                self.lines_spin.setValue(lines)
                self.sequences_spin.setValue(1)
                self._update_accumulation_widgets()
                if self.accu_spinboxes:
                    self.accu_spinboxes[0].setValue(accum)
        except Exception as e:
            self._log_line(f"Markers error: {e}")
            self._log_commit()

    def _update_accumulation_widgets(self):
        while self.accu_container_layout.count():
            item = self.accu_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.accu_spinboxes.clear()

        for i in range(self.sequences_spin.value()):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(QLabel(f"S{i+1}"))
            spin = QSpinBox()
            spin.setRange(1, 1000)
            spin.setValue(1)
            row_layout.addWidget(spin)
            row_layout.addStretch()
            self.accu_container_layout.addWidget(row)
            self.accu_spinboxes.append(spin)

        self.accu_container_layout.addStretch()

    def _on_estimate_shift(self):
        if not self.ptu_data:
            QMessageBox.warning(
                self, "No File", "Please load a PTU file first."
            )
            return
        self.shift_plot_data = None
        self.plot_shift_btn.setEnabled(False)
        self.estimate_btn.setText("Estimating...")
        self._log_start()
        self._log_line("Estimating bidirectional shift...")
        QApplication.processEvents()

        try:
            accumulations = tuple(s.value() for s in self.accu_spinboxes) or (
                1,
            )
            config = ScanConfig(
                lines=self.lines_spin.value(),
                pixels=self.pixels_spin.value(),
                bidirectional=True,
                bidirectional_phase_shift=self.bidir_phase_spin.value(),
                line_accumulations=accumulations,
                max_detector=self.max_detector_spin.value(),
                frame_start_marker_channel=4,
                line_start_marker_channel=1,
                line_stop_marker_channel=2,
            )
            best_shift, plot_data = estimate_bidirectional_shift(
                reader=self.ptu_data["reader"],
                config=config,
                wrap=self.ptu_data["constants"]["wrap"],
                verbose=False,
            )
            if best_shift is not None:
                self.bidir_phase_spin.setValue(best_shift)
                self._log_line(f"Best shift: {best_shift:.5f}")
                self.shift_plot_data = plot_data
                self.plot_shift_btn.setEnabled(True)
            else:
                self._log_line("Estimation failed — check parameters.")
        except Exception as e:
            self._log_line(f"Error: {e}\n{traceback.format_exc()}")
        finally:
            self._log_commit()
            self.estimate_btn.setText("Est.")

    def _on_plot_shift(self):
        if self.shift_plot_data is None:
            return
        dlg = QDialog(self)
        dlg.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        dlg.setWindowTitle("Bidirectional Shift Estimation")
        dlg.resize(420, 300)
        layout = QVBoxLayout(dlg)
        # Compact, dark-themed figure that fits the dialog (smaller fonts,
        # thinner lines, tight_layout so the axes/labels aren't clipped).
        fig = Figure(figsize=(4.0, 2.6), dpi=100, facecolor=MPL.FIG_BG)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        ax = fig.subplots()
        ax.set_facecolor(MPL.AXES_BG)
        shifts, scores, fit = self.shift_plot_data
        ax.plot(shifts, scores, "o-", ms=3, lw=1, label="correlation")
        ax.plot(shifts, fit, "--", lw=1, label="Gaussian fit")
        ax.axvline(
            self.bidir_phase_spin.value(),
            color="r",
            linestyle=":",
            lw=1,
            label=f"best={self.bidir_phase_spin.value():.5f}",
        )
        ax.set_xlabel("Phase shift", fontsize=8, color=MPL.TICK)
        ax.set_ylabel("Score", fontsize=8, color=MPL.TICK)
        ax.tick_params(labelsize=7, colors=MPL.TICK)
        ax.tick_params(axis="x", labelrotation=45)
        for spine in ax.spines.values():
            spine.set_color(MPL.SPINE)
        ax.legend(fontsize=7, labelcolor=MPL.TICK, facecolor=MPL.AXES_BG)
        fig.tight_layout()
        canvas.draw()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()

    def _get_outputs(self) -> list:
        idx = self.out_combo.currentIndex()
        if idx == 0:
            return ["photon_count"]
        if idx == 1:
            return ["photon_count", "mean_arrival_time"]
        return None  # all

    def _on_save_config_json(self):
        """Read the UI values, delegate serialisation to core, write the file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save scan config", "scan_config.json", "JSON (*.json)"
        )
        if not path:
            return
        cfg = build_scan_config_dict(
            frames=self.frames_spin.value(),
            lines=self.lines_spin.value(),
            pixels=self.pixels_spin.value(),
            sequences=self.sequences_spin.value(),
            accumulations=[s.value() for s in self.accu_spinboxes] or [1],
            max_detector=self.max_detector_spin.value(),
            tcspc_bins=self.tcspc_bins_spin.value(),
            bidirectional=self.bidir_group.isChecked(),
            bidirectional_phase_shift=self.bidir_phase_spin.value(),
            f_rep_mhz=float(self.frep_spin.value()),
            factor=_fmt_factor(self.state.calib_factor),
            ptu_filename=(
                self.ptu_filepath.name if self.ptu_filepath else None
            ),
        )
        try:
            save_config(path, cfg)
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _apply_config(self, cfg: dict, source: str = provenance.USER):
        """Apply a scan-config dict (core schema) to the widgets (UI-only).

        Only the fields actually present are applied — anything missing is left
        as-is — and applied fields get their provenance dot set to *source*.
        Any ``ptu_filename`` is ignored here; loading a file is the caller's job.
        """
        scan = cfg.get("scan", {})
        cal = cfg.get("calibration", {})
        # Guard so the programmatic setValue()s don't mark fields as user-edited
        # mid-apply; provenance is set explicitly (below) for applied fields.
        self._autofill = True
        try:
            spins = {
                "lines": self.lines_spin,
                "pixels": self.pixels_spin,
                "frames": self.frames_spin,
                "max_detector": self.max_detector_spin,
                "sequences": self.sequences_spin,
                "tcspc_bins": self.tcspc_bins_spin,
            }
            applied = []
            for key, spin in spins.items():
                if key in scan:
                    spin.setValue(int(scan[key]))
                    applied.append(key)

            # Rebuild accumulation spinboxes for the (new) sequence count, then fill
            self._update_accumulation_widgets()
            for i, acc in enumerate(scan.get("accumulations", [])):
                if i < len(self.accu_spinboxes):
                    self.accu_spinboxes[i].setValue(int(acc))

            self.bidir_group.setChecked(bool(scan.get("bidirectional", False)))
            if "bidirectional_phase_shift" in scan:
                self.bidir_phase_spin.setValue(
                    float(scan["bidirectional_phase_shift"])
                )
            if "f_rep_mhz" in cal:
                self.frep_spin.setValue(float(cal["f_rep_mhz"]))
                applied.append("frep")
            # No widget for the factor here — it is handed to the Phasor tab
            # (via the shared state) once reconstruction finishes.
            if cal.get("factor"):
                self._cfg_calib_factor = complex(cal["factor"])

            for name in applied:
                self._set_param_source(name, source)
        finally:
            self._autofill = False

    def _on_load_config_json(self):
        """Load a config via core, then apply the values to the widgets."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load scan config", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            cfg = load_config(path)
            self._apply_config(cfg, source=provenance.USER)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return
        self._log(f"Loaded config: {Path(path).name}")

    def _build_scan_config(self) -> ScanConfig:
        accumulations = tuple(s.value() for s in self.accu_spinboxes) or (1,)
        return ScanConfig(
            lines=self.lines_spin.value(),
            pixels=self.pixels_spin.value(),
            frames=self.frames_spin.value(),
            line_accumulations=accumulations,
            bidirectional=self.bidir_group.isChecked(),
            bidirectional_phase_shift=self.bidir_phase_spin.value(),
            max_detector=self.max_detector_spin.value(),
            frame_start_marker_channel=4,
            line_start_marker_channel=1,
            line_stop_marker_channel=2,
        )

    def _run_reconstruction(self):
        if not self.ptu_data:
            QMessageBox.warning(
                self, "No Data", "Please load a PTU file first."
            )
            return
        self.log_text.clear()
        self.reconstruct_btn.setEnabled(False)
        self.recon_progress.setValue(0)
        self.recon_progress.setVisible(True)
        self._log("Starting reconstruction...")
        QApplication.processEvents()

        scan_config = self._build_scan_config()
        outputs = self._get_outputs()
        tcspc_override = self.tcspc_bins_spin.value()
        chunk_size = self.chunk_size_spin.value()
        ptu_data = self.ptu_data
        logger = ProgressLogger(mode="collect")
        # Emitting the signal is thread-safe; the slot runs on the main thread.
        emit_progress = self.reconstruction_progress.emit

        def _run():
            return reconstruct_ptu_to_dataset(
                ptu_data=ptu_data,
                scan_config=scan_config,
                outputs=outputs,
                tcspc_channels_override=tcspc_override,
                logger=logger,
                progress_callback=emit_progress,
                chunk_size=chunk_size,
            )

        worker = Worker(_run)
        worker.signals.result.connect(
            lambda ds: self._on_reconstruction_result(ds, scan_config)
        )
        worker.signals.error.connect(self._on_reconstruction_error)
        worker.signals.finished.connect(self._reconstruction_cleanup)
        self.threadpool.start(worker)

    def _reconstruction_cleanup(self):
        self.reconstruct_btn.setEnabled(True)
        self.recon_progress.setVisible(False)

    @Slot(int, int)
    def _on_reconstruction_progress(self, done: int, total: int):
        if total > 0:
            self.recon_progress.setRange(0, total)
            self.recon_progress.setValue(done)
            self.recon_progress.setFormat(f"chunk {done}/{total}")
        else:
            # Unknown record count → indeterminate "busy" bar.
            self.recon_progress.setRange(0, 0)
            self.recon_progress.setFormat(f"chunk {done}")

    @Slot(object)
    def _on_reconstruction_result(self, ds, scan_config):
        constants = self.ptu_data["constants"].copy() if self.ptu_data else {}
        constants["tcspc_bins"] = self.tcspc_bins_spin.value()
        ds.attrs["instrument_params"] = constants
        if scan_config is not None:
            ds.attrs["scan_config"] = scan_config.to_dict()
        if self.ptu_filepath:
            ds.attrs["source_filename"] = self.ptu_filepath.name

        self._log(f"Done. Dataset: {dict(ds.sizes)}")
        if self._cfg_calib_factor is not None:
            self.state.set_calib_factor(self._cfg_calib_factor)
            self._log(f"Calibration factor: {self._cfg_calib_factor}")
        self.state.set_dataset(ds, constants)
        self.reconstruction_finished.emit(ds)

    @Slot(tuple)
    def _on_reconstruction_error(self, error_tuple):
        exctype, value, tb = error_tuple
        self._log(f"ERROR: {value}\n{tb}")
        QMessageBox.critical(self, "Reconstruction Error", str(value))
