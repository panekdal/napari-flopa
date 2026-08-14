import numpy as np
from tttrkit.ptuio.reader import TTTRReader
from tttrkit.ptuio.utils import estimate_tcspc_bins

from napari_flopa.core import provenance
from napari_flopa.core.io.ptu_params import TAG_PARAMS
from napari_flopa.core.logger import ProgressLogger


def format_ptu_header(
    header_tags: dict,
    constants: dict,
    full_header: bool = False,
    constants_source: dict = None,
) -> str:
    """
    Generates a formatted string summary of PTU header and constants.

    Args:
        header_tags: The dictionary of header tags from the PTU file.
        constants: The dictionary of calculated constants.
        full_header: If True, appends the entire raw header dump.
        constants_source: Optional {name: 'metadata'|'default'|'user'|'estimated'} map. When
            given, each key parameter is annotated with an [M]/[D]/[U]/[E] letter.

    Returns:
        A formatted, multi-line string with the summary.
    """
    lines = []

    def _tag(name: str) -> str:
        if constants_source and name in constants_source:
            return f"  [{provenance.letter(constants_source[name])}]"
        return ""

    measurement_sub_mode = header_tags.get("Measurement_SubMode")
    if measurement_sub_mode is not None and measurement_sub_mode < 1:
        lines.append("* WARNING: Not an image. Configure scanning settings.")

    lines += [
        "--- Key Parameters ---",
        f"Repetition Rate:   {constants['repetition_rate']:.2e} Hz{_tag('repetition_rate')}",
        f"TCSPC Resolution:  {constants['tcspc_resolution_ns']:.2e}{_tag('tcspc_resolution')}",
        f"Resolution Unit:   {constants['resolution_unit']}",
        f"TCSPC Bins:        {constants['tcspc_bins']}{_tag('tcspc_bins')}",
        f"Wrap Around:       {constants['wrap']}{_tag('wrap')}",
        f"Omega:             {constants['omega']:.4e} rad/s{_tag('omega')}",
        "",
        "--- Image Header ---",
        f"Pixels X:          {constants.get('pixels_x') or 'N/A'}{_tag('pixels_x')}",
        f"Pixels Y:          {constants.get('pixels_y') or 'N/A'}{_tag('pixels_y')}",
        f"Frame Count:       {constants.get('frames') or 'N/A'}{_tag('frames')}",
    ]

    if full_header:
        lines += ["", "--- Full Header ---"]
        for key, value in header_tags.items():
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def read_ptu_file(
    path, header: bool = True, logger: ProgressLogger = None
) -> dict:
    """
    Reads a PTU file and creates a standardized dictionary of instrument constants.

    Args:
        path: Path to the .ptu file.
        header: If True, includes the full raw header in the log output.
        logger: Optional ProgressLogger. Defaults to print mode.

    Returns:
        dict with keys: 'reader', 'header', 'constants', 'constants_source'
        (the {name: provenance} map for the constants; see ptu_params.py).
    """
    if logger is None:
        logger = ProgressLogger(mode="print")

    logger.log(f"Reading PTU file: {path}")
    reader = TTTRReader(path)
    header_tags = reader.header.tags

    # Read every declared tag param → value + provenance (see ptu_params.py).
    constants, constants_source = {}, {}
    for p in TAG_PARAMS:
        present = p.tag in header_tags
        constants[p.name] = (
            p.transform(header_tags[p.tag]) if present else p.default
        )
        constants_source[p.name] = (
            provenance.METADATA if present else provenance.DEFAULT
        )

    # ── Derived constants (computed from the tag params above) ──────────────
    rep_src = constants_source["repetition_rate"]
    res_src = constants_source["tcspc_resolution"]
    tcspc_resolution = constants["tcspc_resolution"]

    # MeasDesc_Resolution is always in seconds when present (PTU has no unit)
    # the unit is 'ns' when the resolution came from the file, else 'ch'
    constants["tcspc_resolution_ns"] = tcspc_resolution * 1e9
    constants_source["tcspc_resolution_ns"] = res_src
    constants["resolution_unit"] = (
        "ns" if res_src == provenance.METADATA else "ch"
    )
    constants_source["resolution_unit"] = res_src

    # omega 'metadata' only when BOTH inputs came from the header.
    constants["omega"] = (
        2 * np.pi * constants["repetition_rate"] * tcspc_resolution
    )
    constants_source["omega"] = (
        provenance.METADATA
        if rep_src == provenance.METADATA and res_src == provenance.METADATA
        else provenance.DEFAULT
    )

    # buffer = spare channels, user can still override
    # the final count via the "TCSPC Bins" field (tcspc_channels_override)
    constants["tcspc_bins"] = estimate_tcspc_bins(header_tags, buffer=10)
    constants_source["tcspc_bins"] = provenance.ESTIMATED

    summary_text = format_ptu_header(
        header_tags,
        constants,
        full_header=header,
        constants_source=constants_source,
    )
    logger.log(summary_text)

    return {
        "reader": reader,
        "header": header_tags,
        "constants": constants,
        "constants_source": constants_source,
    }
