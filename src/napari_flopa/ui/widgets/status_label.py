"""
Single-line panel status label with three levels.

Every panel that carries a status line uses this instead of styling a bare
QLabel, so the colours and the "! " warning marker stay identical everywhere::

    self._status = StatusLabel("Load and reconstruct a PTU file.")
    ...
    self._status.info(f"Plotted {n:,} pixels.")
    self._status.warn(f"Only {n:,} of {total:,} pixels drawn.")   # amber, "! "
    self._status.error(f"Save error: {e}")                        # red

Each call re-applies its level's stylesheet, so a message never inherits the
previous one's colour.
"""

from qtpy.QtWidgets import QLabel

from napari_flopa.ui.style import S

#: Prefix prepended by warn(); kept here so it cannot drift between panels.
WARN_PREFIX = "! "


class StatusLabel(QLabel):
    """Word-wrapped status line; ``info`` / ``warn`` / ``error`` set the text."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setStyleSheet(S.STATUS)

    def info(self, text: str):
        """Neutral message — dim grey."""
        self._show(text, S.STATUS)

    def warn(self, text: str):
        """Something worth noticing but not a failure — amber, "! " prefixed."""
        self._show(
            text if text.startswith(WARN_PREFIX) else WARN_PREFIX + text,
            S.STATUS_WARN,
        )

    def error(self, text: str):
        """An operation failed — red."""
        self._show(text, S.STATUS_ERROR)

    def _show(self, text: str, style: str):
        self.setStyleSheet(style)
        self.setText(text)
