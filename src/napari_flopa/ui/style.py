"""
Centralised visual style tokens for napari-flopa widgets.

Usage
-----
    from napari_flopa.ui.style import C, S, MPL, apply_style

    apply_style(group_box, S.GROUP_PRIMARY)
    apply_style(label, S.STATUS)
    apply_style(btn, S.BTN_DANGER)
    ax.set_facecolor(MPL.AXES_BG)

All Qt stylesheet strings are module-level constants on ``S``.
All raw hex colours are on ``C``.
Matplotlib plot colours are on ``MPL``.

Group box title variants
------------------------
  S.GROUP_PRIMARY — primary sections; gold title.
                Supports the ``#plain`` object-name selector for a muted
                gray title: ``box.setObjectName("plain")``.
  S.GROUP_NESTED  — secondary / nested sections; amber title.
  S.GROUP_DOCK    — top-level dock container (e.g. FLIM View); teal title.
  S.GROUP_COMPACT — bordered compact box used in dense panel layouts.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Colour tokens
# ──────────────────────────────────────────────────────────────────────────────


class C:  # "Color" — raw hex colour tokens
    """Raw hex colour tokens — single source of truth for the dark theme.

    Tuned to sit inside napari's dark blue-grey theme (background #262930,
    foreground #414851, text #f0f1f2). Token *names* are the public contract
    (S/MPL build on them); only the values changed.
    """

    # Surfaces (dark → light), matching napari's blue-greys
    BG_DEEP = "#1c1e24"  # deepest background (log, console)
    BG_DARK = "#262930"  # panel / plot background (napari `background`)
    BG_MID = "#2e323b"  # input field / axes background
    BG_RAISED = (
        "#3d444e"  # slider groove, raised surface (~napari `foreground`)
    )
    BG_SECTION = "#2b323c"  # tinted group-box background (section)

    # Borders / separators
    BORDER = "#3a414b"
    BORDER_SOFT = "#31363e"
    BORDER_DEFAULT = "#565e68"  # ~napari `primary`

    # Text
    TEXT = "#e4e6e9"  # primary text (~napari `text`)
    TEXT_MUTED = "#b4b9c0"  # secondary / read-only text
    TEXT_DIM = "#868e93"  # status / hint text (napari `secondary`)
    TEXT_FAINT = "#6b727b"  # extra-faint hint
    TEXT_DARK = "#565e68"  # disabled / inactive

    # Accent — cyan/teal (contrast / view slider)
    ACCENT = "#38bec9"
    ACCENT_DIM = "#38bec9"
    ACCENT_BG = "#22343a"
    ACCENT_BG_HOV = "#2b4249"

    # Accent — red (mask / danger)
    DANGER = "#e05656"
    DANGER_DIM = "#a83b3b"
    DANGER_BG = "#3d2226"
    DANGER_TEXT = "#ef6b6b"
    DANGER_SOFT = "#f09393"
    DANGER_DARK = "#6b4a4e"
    DANGER_BG_DIS = "#2a1e21"

    # Accent — (execute / done)
    SUCCES = "#ecff40"
    SUCCES_BG = "#264242"

    # Accent — amber (warning / secondary title)
    WARNING = "#e0a53a"  # warning label
    TITLE = "#d9c45e"  # GROUP_PRIMARY main title (soft gold)
    TITLE_PLAIN = "#c8ccd2"  # GROUP_PRIMARY #plain variant (light gray)
    TITLE_NESTED = "#c9a24a"  # GROUP_NESTED title (amber)

    # Accent - pink/purple
    PINK_LINE = "#FC24DF"

    # Categorical series palette (Okabe–Ito, colour-blind safe) — one entry per
    # detector in the trace plot; cycles if there are more series than colours.
    SERIES = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
    )

    # Stale indicator
    STALE_INACTIVE = "#565e68"
    STALE_STALE = "#e05656"
    STALE_FRESH = "#5fbf74"

    # Parameter provenance (source: metadata / default / user / estimated)
    PROV_METADATA = "#d9c45e"
    PROV_DEFAULT = "#868e93"
    PROV_USER = "#4a90d9"
    PROV_ESTIMATED = "#e0a53a"


# ──────────────────────────────────────────────────────────────────────────────
# Qt stylesheet strings
# ──────────────────────────────────────────────────────────────────────────────


class S:  # "Style" — Qt stylesheet strings
    """Qt stylesheet strings for common widget roles."""

    # ── Labels ────────────────────────────────────────────────────────────────

    # Provenance dot, keyed by source
    PROV_DOT = {
        "metadata": f"color: {C.PROV_METADATA}; font-size: 11px;",
        "default": f"color: {C.PROV_DEFAULT}; font-size: 11px;",
        "user": f"color: {C.PROV_USER}; font-size: 11px;",
        "estimated": f"color: {C.PROV_ESTIMATED}; font-size: 11px;",
    }

    # Status line levels
    STATUS = f"color: {C.TEXT_DIM}; font-size: 11px;"
    STATUS_WARN = f"color: {C.WARNING}; font-size: 10px;"
    STATUS_ERROR = f"color: {C.DANGER_TEXT}; font-size: 10px;"
    HINT = f"color: {C.TEXT_FAINT}; font-size: 9px; font-weight: normal;"
    MUTED = f"color: {C.TEXT_MUTED}; font-weight: normal;"
    WARNING = f"color: {C.WARNING}; font-size: 9px; font-weight: normal;"
    SEPARATOR = f"color: {C.BORDER};"

    # ── Stale indicator (● dot label) ─────────────────────────────────────────

    STALE_INACTIVE = f"color: {C.STALE_INACTIVE}; font-size: 16px;"
    STALE_STALE = f"color: {C.STALE_STALE};    font-size: 16px;"
    STALE_FRESH = f"color: {C.STALE_FRESH};    font-size: 16px;"

    # ── Read-only display ─────────────────────────────────────────────────────

    DISPLAY = f"color: {C.TEXT_MUTED}; font-family: monospace;"

    # ── Buttons ───────────────────────────────────────────────────────────────

    BTN_DANGER = (
        f"QPushButton {{ color: {C.DANGER_TEXT}; }}"
        f"QPushButton:disabled {{ color: {C.DANGER_DIM}; }}"
    )

    BTN_SUCCESS = f"QPushButton {{ color: {C.SUCCES}; }}"

    BTN_RUN = f"QPushButton {{ color: {C.SUCCES}; min-width: 50px;}}"

    BTN_STOP = f"QPushButton {{ color: {C.DANGER_TEXT}; min-width: 50px;}}"

    BTN_SMALL = "font-size: 10px;"

    BTN_DET_ON = (
        f"QPushButton {{ background: {C.ACCENT_BG}; color: {C.ACCENT}; "
        f"border: 1px solid {C.ACCENT_DIM}; border-radius: 3px; "
        f"padding: 1px 6px; font-size: 10px; }}"
        f"QPushButton:hover {{ background: {C.ACCENT_BG_HOV}; }}"
    )
    BTN_DET_OFF = (
        f"QPushButton {{ background: {C.BG_MID}; color: {C.TEXT_DARK}; "
        f"border: 1px solid {C.BORDER}; border-radius: 3px; "
        f"padding: 1px 6px; font-size: 10px; }}"
    )
    BTN_DET_DISABLED = (
        f"QPushButton {{ background: {C.BG_DEEP}; color: {C.TEXT_DARK}; "
        f"border: 1px solid {C.BORDER_SOFT}; border-radius: 3px; "
        f"padding: 1px 6px; font-size: 10px; }}"
    )

    # ── Input fields ──────────────────────────────────────────────────────────

    LINE_EDIT = (
        f"QLineEdit {{ background: {C.BG_MID}; color: {C.TEXT}; "
        f"border: 1px solid {C.BORDER}; border-radius: 2px; "
        f"padding: 1px 2px; font-size: 9px; }}"
        f"QLineEdit:focus {{ border: 1px solid #888888; }}"
    )

    # ── Sliders ───────────────────────────────────────────────────────────────

    SLIDER_VIEW = (
        f"QSlider::groove:horizontal {{ background: {C.BG_RAISED}; height: 4px; border-radius: 2px; }}"
        f"QSlider::handle:horizontal {{ background: {C.ACCENT}; width: 10px; height: 10px;"
        f" margin: -3px 0; border-radius: 5px; }}"
        f"QSlider::sub-page:horizontal {{ background: {C.ACCENT_DIM}; border-radius: 2px; }}"
    )

    SLIDER_MASK = (
        f"QSlider::groove:horizontal {{ background: {C.BG_RAISED}; height: 4px; border-radius: 2px; }}"
        f"QSlider::handle:horizontal {{ background: {C.DANGER}; width:10px; height: 10px;"
        f" margin: -3px 0; border-radius: 5px; }}"
        f"QSlider::sub-page:horizontal {{ background: {C.DANGER_DIM}; border-radius: 2px; }}"
    )

    # ── Group boxes ───────────────────────────────────────────────────────────

    GROUP_PRIMARY = f"""
    QGroupBox {{
        margin-top: 14px;
        border: 1px {C.TITLE};
        border-radius: 0px;
        background-color: {C.BG_SECTION};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        padding: 0px 2px;
        font-size: 12pt;
        font-weight: bold;
        color: {C.TITLE};
    }}
    QGroupBox#plain::title {{
        subcontrol-origin: margin;
        padding: 0px 2px;
        font-size: 12pt;
        font-weight: bold;
        color: {C.TITLE_PLAIN};
    }}
    """

    GROUP_PLAIN = f"""
    QGroupBox {{
        margin-top: 0px;
        border: 1px {C.TITLE};
        border-radius: 0px;
        background-color: {C.BG_DARK};
    }}
    """

    GROUP_NESTED = f"""
    QGroupBox {{
        margin-top: 1px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        padding: 0px 2px;
        font-size: 12pt;
        font-weight: bold;
        color: {C.TITLE_NESTED};
    }}
    """

    GROUP_CHECKABLE = f"""
    QGroupBox {{
        background-color: {C.BG_SECTION};
        border: none;
        margin-top: 16px;
        padding-top: 2px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 2px;
        padding: 0px 3px;
        font-size: 11pt;
        color: {C.TITLE_PLAIN};
        background-color: {C.BG_SECTION};
    }}
    """

    GROUP_DOCK = f"""
    QGroupBox {{
        margin-top: 14px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        padding: 0px 2px;
        font-size: 12pt;
        color: {C.ACCENT_DIM};
    }}
    """

    GROUP_COMPACT = f"""
    QGroupBox {{
        border: 1px solid {C.BORDER};
        border-radius: 3px;
        margin-top: 8px;
        padding-top: 4px;
        font-weight: bold;
        color: {C.TEXT};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
    }}
    """

    # ── Log / console / header views ──────────────────────────────────────────

    LOG = (
        f"QPlainTextEdit, QTextEdit {{ background: {C.BG_DEEP}; color: {C.TEXT_MUTED}; "
        f"border: 1px solid {C.BORDER_SOFT}; font-family: Courier, monospace; "
        f"font-size: 8pt; }}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────


def apply_style(widget, style_string: str) -> None:
    """Apply a Qt stylesheet string to *widget*. Convenience wrapper."""
    widget.setStyleSheet(style_string)


# ──────────────────────────────────────────────────────────────────────────────
# Plot theme
# ──────────────────────────────────────────────────────────────────────────────


class MPL:  # "Plot"
    """Colour values for figure/axes styling (not Qt stylesheets)."""

    FIG_BG = C.BG_DARK  # figure.facecolor
    AXES_BG = C.BG_MID  # axes.facecolor
    TICK = C.TEXT  # tick label colour
    SPINE = C.TEXT_DARK  # axes spine colour
    GRID = "#333842"  # grid line colour

    # Dragged-range overlay in the trace plot
    SELECTION = C.TITLE
    SELECTION_FILL_ALPHA = 30
    SELECTION_EDGE_ALPHA = 160

    # Marker overlays in the trace plot.
    MARKER_LINE_START = C.STALE_FRESH
    MARKER_LINE_STOP = C.STALE_STALE
    MARKER_FRAME = C.PINK_LINE

    # Plate behind legend text.
    LEGEND_BG = C.BG_DARK
    LEGEND_BG_ALPHA = 245
