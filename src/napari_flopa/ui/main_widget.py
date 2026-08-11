import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QScrollArea, QTabWidget, QVBoxLayout, QWidget

from napari_flopa.ui.panels.batch_panel import BatchPanel
from napari_flopa.ui.panels.decay_panel import DecayPanel
from napari_flopa.ui.panels.flim_view_panel import FlimViewPanel
from napari_flopa.ui.panels.phasor_panel import PhasorPanel
from napari_flopa.ui.panels.ptu_panel import PtuPanel
from napari_flopa.ui.state import FlopaState


def _scrollable(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    return scroll


class FlimWidget(QWidget):
    """
    Main napari plugin widget. Tab container with panels:
      0 — File
      1 — Phasor
      2 — Decay
      3 - Batch

    On first reconstruction, a FlimViewPanel is added as a bottom dock widget.
    When the FLIM View selection changes, Phasor and Decay panels are notified
    so they can show their stale indicator.
    """

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.state = FlopaState()
        self._view_dock = None
        self._view_panel = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # self.setMinimumWidth(80)

        self._tabs = QTabWidget()
        self._ptu_panel = PtuPanel(self.state, viewer)
        self._phasor_panel = PhasorPanel(self.state, viewer)
        self._decay_panel = DecayPanel(self.state, viewer)
        self._batch_panel = BatchPanel(self.state, viewer)

        self._tabs.addTab(_scrollable(self._ptu_panel), "File")
        self._tabs.addTab(_scrollable(self._phasor_panel), "Phasor")
        self._tabs.addTab(_scrollable(self._decay_panel), "Decay")
        self._tabs.addTab(_scrollable(self._batch_panel), "Batch")

        # Phasor/Decay tabs start disabled until data is available
        self._tabs.setTabEnabled(1, False)
        self._tabs.setTabEnabled(2, False)
        self._tabs.setTabEnabled(3, True)

        layout.addWidget(self._tabs)

        self._ptu_panel.reconstruction_finished.connect(
            self._on_reconstruction_finished
        )
        self.state.dataset_changed.connect(self._on_dataset_changed)

    def _on_reconstruction_finished(self, ds):
        self._add_view_panel(ds)

    def _on_dataset_changed(self):
        ds = self.state.dataset
        has_phasor = ds is not None and "phasor_g" in ds and "phasor_s" in ds
        has_tcspc = ds is not None and "tcspc_histogram" in ds
        self._tabs.setTabEnabled(1, has_phasor)
        self._tabs.setTabEnabled(2, has_tcspc)

    def _add_view_panel(self, ds):
        """Add FlimViewPanel as a bottom dock widget (once only)."""
        if self._view_dock is None:
            self._view_panel = FlimViewPanel(self.state, self.viewer)
            self._view_dock = self.viewer.window.add_dock_widget(
                self._view_panel,
                name="FLIM View",
                area="bottom",
            )
            # Wire FLIM View selection changes → stale indicators in analysis tabs
            self._view_panel.view_changed.connect(
                self._phasor_panel.on_view_changed
            )
            self._view_panel.view_changed.connect(
                self._decay_panel.on_view_changed
            )

        self._view_panel.update_data(ds)
        self._view_dock.setVisible(True)
