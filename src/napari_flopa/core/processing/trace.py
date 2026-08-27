"""
Intensity trace reconstruction: photon counts binned over time.

Unlike image reconstruction this ignores scan geometry entirely — it bins every
photon of a time window into fixed-width bins per detector, and collects the
marker times that fall inside the same window. Used by the Trace tab to inspect
what a measurement actually looked like over time.
"""

import xarray as xr
from tttrkit.ptuio.decoder import T3OverflowCorrector
from tttrkit.ptuio.reconstructor import TraceReconstructor

from napari_flopa.core.io.loader import read_ptu_file
from napari_flopa.core.logger import ProgressLogger
from napari_flopa.core.processing.reconstruction import DEFAULT_CHUNK_SIZE


def reconstruct_trace(
    ptu_path,
    *,
    start_time: float,
    stop_time: float,
    max_detector: int,
    bin_width_s: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    logger: ProgressLogger | None = None,
    progress_callback=None,
) -> xr.Dataset:
    """Bin photons into a time trace over ``[start_time, stop_time]`` seconds.

    Args:
        ptu_path: Path to the .ptu file.
        start_time, stop_time: Window in seconds; only this part is read.
        max_detector: Highest detector index to bin.
        bin_width_s: Time bin width in **seconds**.
        chunk_size: TTTR records per iteration — throughput only.
        logger: Optional ProgressLogger.
        progress_callback: Optional ``callable(done_chunks)`` after each chunk.

    Returns:
        xr.Dataset with ``photon_count`` (time, channel) and the marker times
        ``frame_start_times`` / ``line_start_times`` / ``line_stop_times``,
        all in seconds.

    Raises:
        ValueError: If the window is empty (delegated to TraceReconstructor).
    """
    if logger is None:
        logger = ProgressLogger(mode="print")

    ptu_data = read_ptu_file(str(ptu_path), header=False)
    reader = ptu_data["reader"]
    constants = ptu_data["constants"]
    sync_rate = float(constants["repetition_rate"])

    recon = TraceReconstructor(
        start_time=start_time,
        stop_time=stop_time,
        max_detector=max_detector,
        bin_width=bin_width_s,
        sync_rate=sync_rate,
        outputs=["photon_count", "markers"],
    )
    corrector = T3OverflowCorrector(wraparound=constants["wrap"])

    # Records are time-ordered and `nsync` is overflow-corrected, so once a
    # chunk ends past the window nothing later can fall inside it. Reading a
    # short window out of a long acquisition therefore costs a few chunks
    # rather than the whole file.
    stop_nsync = stop_time * sync_rate

    for chunk_num, chunk in enumerate(
        reader.iter_chunks(chunk_size=chunk_size), start=1
    ):
        corrected = corrector.correct(chunk)
        recon.update(corrected)
        logger.log(f"Processed chunk {chunk_num}...")
        if progress_callback is not None:
            progress_callback(chunk_num)
        if corrected.size and corrected["nsync"][-1] > stop_nsync:
            logger.log(f"Reached {stop_time} s after {chunk_num} chunk(s).")
            break

    logger.log("Finalizing trace...")
    return recon.finalize()
