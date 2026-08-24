import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.signal import convolve2d


def smooth_weighted(
    array: NDArray[np.floating],
    count: NDArray[np.integer],
    size: int = 3,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """
    Apply photon-weighted box smoothing to a real-valued 2-D scalar field.

    Smooths `array` using a uniform square kernel of side `size`, weighting
    each pixel by `count`.  Invalid entries (NaN, Inf, or zero/negative count)
    are excluded from both numerator and denominator so they do not bleed into
    neighbouring pixels.  Output pixels with no valid kernel contribution are
    set to NaN.

    Intended for scalar fields such as mean_arrival_time.  Do NOT use this
    for phasor data — use smooth_phasor from tttrkit.analysis.phasor instead, which operates
    on the complex g+i·s representation directly and handles both components
    in a single pass.
    """
    array = np.asarray(array)
    count = np.asarray(count)
    if array.ndim != 2 or count.ndim != 2:
        raise ValueError("array and count must both be 2D arrays")
    assert (
        array.shape == count.shape
    ), "array and count must have the same shape"
    if not (isinstance(size, int) and size > 0):
        raise ValueError("kernel size must be a positive integer")
    if size % 2 == 0:  # even kernel has no centre → mode="same" shifts by ½px
        size += 1
    kernel = np.ones((size, size), dtype=np.float32)
    valid = np.isfinite(array) & (count > 0)
    num = convolve2d(
        np.where(valid, array * count, 0).astype(np.float32),
        kernel,
        mode="same",
    )
    den = convolve2d(
        np.where(valid, count, 0).astype(np.float32), kernel, mode="same"
    )
    out = np.full_like(array, np.nan, dtype=np.float32)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out, np.asarray(den, dtype=np.uint32)


def smooth_count(
    count: NDArray[np.integer], size: int = 3
) -> NDArray[np.float32]:
    """
    Edge-corrected box-*mean* smoothing of a photon-count array.

    Each window is normalised by the number of in-bounds pixels, so border
    pixels are NOT darkened by zero-padding — the same den-normalisation that
    smooth_weighted (and tttrkit's smooth_phasor) use. Returns a float32 local
    average count; display contrast is handled by the histogram slider.
    """
    count = np.asarray(count, dtype=np.float32)
    if count.ndim != 2:
        raise ValueError("count must be a 2D array")
    if size % 2 == 0:  # keep the kernel centred (odd); even shifts by ½px
        size += 1
    kernel = np.ones((size, size), dtype=np.float32)
    total = convolve2d(count, kernel, mode="same")
    norm = convolve2d(np.ones_like(count), kernel, mode="same")
    return np.asarray(total / norm, dtype=np.float32)


def aggregate_dataset(ds: xr.Dataset, dims) -> xr.Dataset:
    if isinstance(dims, str):
        dims = [dims]
    if not dims:
        return ds
    missing = [d for d in dims if d not in ds.sizes]
    if missing:
        raise ValueError(f"Dims not in dataset: {missing}")

    out = {}
    photon_sum = None
    if "photon_count" in ds:
        photon_sum = ds["photon_count"].sum(dim=dims, keepdims=True)
        out["photon_count"] = photon_sum.astype("uint64")

    for var in ["mean_arrival_time", "phasor_g", "phasor_s"]:
        if var in ds and photon_sum is not None:
            valid = ds[var].notnull()
            num = (ds[var].where(valid, 0) * ds["photon_count"]).sum(
                dim=dims, keepdims=True
            )
            den = (
                ds["photon_count"].where(valid, 0).sum(dim=dims, keepdims=True)
            )
            out[var] = xr.where(photon_sum > 0, num / den, np.nan).astype(
                "float32"
            )

    if "tcspc_histogram" in ds:
        # tcspc_histogram is a global decay (frame, channel, tcspc_channel) and
        # does not carry the spatial dims (sequence/line/pixel) the other
        # variables do — only reduce over the dims it actually has.
        hist = ds["tcspc_histogram"]
        hist_dims = [d for d in dims if d in hist.dims]
        out["tcspc_histogram"] = (
            hist.sum(dim=hist_dims, keepdims=True).astype("uint64")
            if hist_dims
            else hist
        )

    if not out:
        return ds

    out_ds = xr.Dataset(out)
    coord_ds = ds.coords.to_dataset()
    indexers = {d: slice(0, 1) for d in dims if d in coord_ds.sizes}
    coord_ds = coord_ds.isel(indexers)

    # Ensure every reduced dimension has a dimension coordinate, even when
    # the original dataset had no explicit coordinate for that dim.  Without
    # this, the merged result can have size-1 dims with no coordinate, which
    # breaks downstream isel / sel calls that expect a coordinate to exist.
    for d in dims:
        if d in ds.dims and d not in coord_ds:
            if d in ds.coords:
                coord_ds[d] = ds[d].isel({d: slice(0, 1)})
            else:
                coord_ds[d] = xr.DataArray(np.arange(1), dims=(d,))

    return xr.merge([out_ds, coord_ds], compat="override")


#: Colormaps offered wherever the UI lets one be picked (matplotlib names).
#: Kept here, next to `colormap_to_lut`, so the FLIM View and Batch tabs cannot
#: drift apart on what is selectable.
LIFETIME_COLORMAPS = ("rainbow", "hsv", "viridis", "magma", "cividis")
INTENSITY_COLORMAPS = ("gray", "viridis", "magma", "hot", "cividis")


def colormap_to_lut(cmap_name: str, n: int = 256) -> NDArray[np.uint8]:
    """Build an (n, 3) uint8 RGB lookup table from a matplotlib colormap name.

    Pre-building the LUT once lets `flim_rgb` recolour interactively (a uint8
    index lookup) instead of evaluating the colormap on every slider move.
    """
    # `matplotlib.colormaps[name]` is the registry lookup that replaced
    # `cm.get_cmap()` (deprecated in 3.7, removed in 3.11).
    from matplotlib import colormaps

    cmap = colormaps[cmap_name]
    return (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)


def flim_export_info(
    cmap: str,
    lt_range: tuple[float, float],
    int_range: tuple[float, float],
    unit: str = "ch",
) -> str:
    """Sidecar text describing how a FLIM RGB image was composited.

    An exported RGB is not quantitative on its own — these are the numbers
    needed to read it back. Shared by the interactive and batch exports so the
    two sidecars stay comparable.
    """
    lt_lo, lt_hi = lt_range
    int_lo, int_hi = int_range
    return (
        "FLIM RGB export\n"
        f"lifetime colormap : {cmap}\n"
        f"lifetime range    : {lt_lo:.4g} .. {lt_hi:.4g} {unit}\n"
        f"intensity range   : {int_lo:.4g} .. {int_hi:.4g} photon counts\n"
    )


def auto_range(
    values: NDArray[np.floating], mode: str = "minmax"
) -> tuple[float, float]:
    """Display range derived from the data itself, ignoring NaN/Inf.

    ``minmax`` uses the extremes of *values*; ``p2p98`` the 2nd/98th
    percentiles. The mode argument exists so callers can offer more options
    later without changing their call sites. Falls back to ``(0.0, 1.0)`` when
    nothing is finite.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    if mode == "p2p98":
        lo, hi = np.percentile(finite, [2, 98])
    elif mode == "minmax":
        lo, hi = finite.min(), finite.max()
    else:
        raise ValueError(f"Unknown auto-range mode: {mode!r}")
    return float(lo), float(hi)


def flim_rgb(
    intensity: NDArray[np.floating],
    lifetime: NDArray[np.floating],
    lut: NDArray[np.uint8],
    lt_range: tuple[float, float],
    int_range: tuple[float, float],
) -> NDArray[np.float32]:
    """Composite a FLIM RGB image: lifetime → colour (via ``lut``) × intensity.

    Lifetime is normalised into ``lt_range`` and quantised to ``len(lut)`` colour
    bins; intensity is normalised into ``int_range`` (0..1) and scales the colour.
    NaN lifetimes map to the low end of ``lt_range``; NaN intensities map to 0.

    Returns an (H, W, 3) float32 array in [0, 1]. This is the single source of
    truth for both the interactive display and the exported image, so the two
    always match (and it is equivalent to a quantised tttrkit ``create_FLIM_image``).
    """
    lt_lo, lt_hi = lt_range
    span = lt_hi - lt_lo if lt_hi > lt_lo else 1.0
    lt_f = np.where(np.isfinite(lifetime), lifetime, lt_lo).astype(np.float32)
    n = lut.shape[0]
    lt_idx = np.clip((lt_f - lt_lo) / span * (n - 1), 0, n - 1).astype(
        np.uint16
    )

    int_lo, int_hi = int_range
    int_span = int_hi - int_lo if int_hi > int_lo else 1.0
    int_f = np.where(np.isfinite(intensity), intensity, int_lo).astype(
        np.float32
    )
    int_norm = np.clip((int_f - int_lo) / int_span, 0, 1)

    return (lut[lt_idx].astype(np.float32) / 255.0) * int_norm[..., np.newaxis]
