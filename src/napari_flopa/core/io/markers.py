import numpy as np
from tttrkit.ptuio.decoder import T3OverflowCorrector
from tttrkit.ptuio.marker import get_marker_distribution, marker_events
from tttrkit.ptuio.reader import TTTRReader

from napari_flopa.core.io.ptu_params import read_tag


def get_markers(reader: TTTRReader, chunk_limit: int = 0) -> dict:
    """
    Reads chunks from a PTU file and extracts the distribution of markers.

    Args:
        reader: An initialized TTTRReader for the PTU file.
        chunk_limit: Number of 1M-record chunks to read (0 = all).

    Returns:
        A dict mapping marker channel numbers to their counts,
        or {"error": "..."} if no markers found.
    """
    all_markers = []
    wrap = read_tag(reader.header.tags, "wrap")
    corrector = T3OverflowCorrector(wraparound=wrap)

    for i, chunk in enumerate(reader.iter_chunks(chunk_size=1_000_000)):
        if chunk_limit > 0 and i >= chunk_limit:
            break
        corrected_chunk = corrector.correct(chunk)
        all_markers.append(marker_events(corrected_chunk))

    all_markers_flat = np.concatenate(all_markers)
    if all_markers_flat.size == 0:
        return {"error": "No markers found."}

    return get_marker_distribution(all_markers_flat)


def analyze_marker_distribution(
    distribution: dict,
    verbose: bool = False,
    line_start_marker: int = 1,
    frame_start_marker: int = 4,
    max_accumulations: int = 64,
) -> dict:
    """
    Analyzes a marker distribution to suggest scan parameters.

    Args:
        distribution: Output from get_markers().
        verbose: If True, prints a formatted summary.
        line_start_marker: Marker channel for line starts.
        frame_start_marker: Marker channel for frame starts.
        max_accumulations: Max accumulations to consider in suggestions.

    Returns:
        dict with structured analysis results and suggested (lines, accumulations) pairs.
    """
    num_line_starts = distribution.get(line_start_marker, 0)
    num_frame_starts = distribution.get(frame_start_marker, 0)

    frames_guess = max(1, num_frame_starts)
    total_lines_per_frame = num_line_starts // frames_guess

    suggestion_pairs = []
    for i in range(1, max_accumulations + 1):
        if total_lines_per_frame % i == 0:
            lines = total_lines_per_frame // i
            if 64 <= lines <= 4096:
                suggestion_pairs.append((lines, i))

    analysis_results = {
        "num_line_starts": num_line_starts,
        "num_frame_starts": num_frame_starts,
        "frames_guess": frames_guess,
        "total_lines_per_frame": total_lines_per_frame,
        "suggestions": suggestion_pairs,
    }

    if verbose:
        print("--- Marker Analysis Suggestions ---")
        print(_format_marker_suggestions(analysis_results))

    return analysis_results


def _format_marker_suggestions(analysis_results: dict) -> str:
    lines = [
        f"Frame Starts: {analysis_results['num_frame_starts']} | Line Starts: {analysis_results['num_line_starts']}",
        f"For {analysis_results['frames_guess']} frame(s) ~ {analysis_results['total_lines_per_frame']} line scans per frame.",
        "",
        "Possible combinations: Lines x Accumulations",
    ]
    suggestions = analysis_results.get("suggestions", [])
    if not suggestions:
        lines.append(
            "  - Could not find common factors. Please check header or lab notes."
        )
    else:
        for lines_val, acc_val in suggestions:
            lines.append(f"  - {lines_val} x {acc_val}")
    return "\n".join(lines)
