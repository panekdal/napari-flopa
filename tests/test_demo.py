"""Tests exercising the bundled demo dataset end-to-end.

These skip automatically when no demo data is available (e.g. on a checkout
where the shipped demo is absent), so they are safe on CI. Locally, a
``demo_params.local.json`` override lets you run them against a larger file.
"""

import pytest

from napari_flopa.core import demo

requires_demo = pytest.mark.skipif(
    demo.demo_params_path() is None,
    reason="no demo data available (see src/napari_flopa/core/demo.py)",
)


def _scan_config(params):
    from tttrkit.ptuio.reconstructor import ScanConfig

    scan = params["scan"]
    return ScanConfig(
        lines=scan["lines"],
        pixels=scan["pixels"],
        frames=scan["frames"],
        line_accumulations=tuple(scan["accumulations"]),
        max_detector=scan["max_detector"],
        bidirectional=scan.get("bidirectional", False),
        bidirectional_phase_shift=scan.get("bidirectional_phase_shift", 0.0),
        frame_start_marker_channel=(4,),
        line_start_marker_channel=(1,),
        line_stop_marker_channel=(2,),
    )


@requires_demo
def test_demo_reconstructs():
    """Core pipeline: read + reconstruct the demo .ptu into a dataset."""
    from napari_flopa.core.io.loader import read_ptu_file
    from napari_flopa.core.processing.reconstruction import (
        reconstruct_ptu_to_dataset,
    )

    params, ptu = demo.load_demo()
    data = read_ptu_file(str(ptu), header=False)
    ds = reconstruct_ptu_to_dataset(
        data, _scan_config(params), outputs=["photon_count"]
    )

    assert "photon_count" in ds
    assert ds["photon_count"].sizes["pixel"] == params["scan"]["pixels"]
    assert ds["photon_count"].sizes["line"] == params["scan"]["lines"]


@requires_demo
def test_load_demo_button(make_napari_viewer):
    """Widget smoke test: the Load Demo button populates state without error."""
    from napari_flopa.ui.main_widget import FlimWidget

    viewer = make_napari_viewer()
    widget = FlimWidget(viewer)
    widget._ptu_panel._on_load_demo()

    assert widget._ptu_panel.ptu_data is not None
