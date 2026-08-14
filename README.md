# napari-flopa

[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue)](https://napari.org/stable/plugins/index.html)
[![PyPI](https://img.shields.io/pypi/v/napari-flopa.svg?color=green)](https://pypi.org/project/napari-flopa)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-flopa.svg?color=green)](https://python.org)


> **Work in progress** — the plugin is functional but under active development. Expect breaking changes between versions.

A [napari] plugin for opening, processing and analysing FLIM (Fluorescence Lifetime Imaging Microscopy) data from `.ptu` files.

## Features

- **Process PTU** — reconstruct `.ptu` files into xarray datasets (photon count, mean arrival time, phasor, TCSPC histogram); supports multi-frame, multi-sequence and multi-detector data
- **FLIM View** — interactive display with histogram contrast sliders for intensity and lifetime; FLIM RGB composite; export to TIFF
- **Phasor** — phasor plot with calibration, smoothing, per-object or per-pixel scatter, monoexponential lifetime semi-circle overlay
- **Decay** — TCSPC decay plot with aggregation, normalisation and log scale
- **Batch** — process a folder of `.ptu` files (opt. with masks) with a shared scan config and export images, phasor tables and decay tables; config saved/loaded as json

## Requirements

- Python ≥ 3.11
- [napari] with a Qt backend

## Installation

Install into the environment where napari runs. Pick one:


**From PyPI**

```bash
pip install napari-flopa            # plugin only, for an existing napari install
pip install "napari-flopa[all]"     # plugin + napari + Qt
```

Use the plain (non-`[all]`) form when you already have napari installed.

**From source**

```bash
git clone https://github.com/cockovaz/napari-flopa
cd napari-flopa
pip install -e ".[all]"      # editable install, napari + Qt included
```

The `napari_flopa.core` package (I/O, reconstruction, image and phasor maths) imports
no GUI libraries, so it can be used from a plain script or notebook without napari.

## Getting started

**1. Open the plugin.** Start napari and choose **Plugins → FLOPA → FLIM Analysis**.

**2. Load a file.** In the **File** tab, click **Load Demo** for the bundled demo
dataset, or **Read PTU…** for your own. The header is parsed and the scan parameters
are filled in; the coloured dot beside each field says where its value came from —
file metadata, a default, an estimate, or your own edit.

**3. Set the scan geometry.** Frames, lines, pixels, sequences and accumulations must match how the image
was actually acquired, because the raw file is a stream of photon and marker events
with no image shape of its own. The header supplies what it knows; fill in the rest.
**Analyze Markers** inspects the marker events and suggests dimensions.

**4. Reconstruct.** Pick what to compute under **Output**:

- *Intensity* — photon-count image only, the fastest
- *Int. + τ* — adds the mean arrival time (lifetime) image
- *All* — adds phasor coordinates and the TCSPC decay (enables the
  **Phasor** and **Decay** tabs)

**5. Look at the result.** The **FLIM View** dock opens at the bottom, with a
histogram for Intensity and one for Lifetime. Each has two sliders: the **cyan** one
sets the display contrast, the **red** one a threshold range. **→ Generate Int./Lt.
Mask** turns that range into a napari Labels layer. Intensity, lifetime and the FLIM RGB composite can be exported from here.

**6. Analyse.** **Phasor** plots g/s per object or per pixel — apply a calibration
factor, pick a Labels layer to colour by object or to restrict the plot to a region.
**Decay** plots the TCSPC curves, with **From View** to follow the frame/detector
currently shown in FLIM View.

**7. Reuse the settings.** Use **Batch** tab to run a whole folder of `.ptu` files with identical settings.

## Data model

Reading and reconstructing `.ptu` files is done by **[tttrkit]** (imported as
`tttrkit.ptuio`), a separate package developed alongside this plugin. It is a
normal dependency and is installed automatically — see [tttrkit] for the raw
TTTR parsing, scan reconstruction and phasor maths that sit underneath the GUI.

Reconstruction produces a single `xarray.Dataset` holding up to five variables.

Four of them are images and share the dimensions
`(frame, sequence, line, pixel, channel)`: `photon_count`, `mean_arrival_time`,
`phasor_g` and `phasor_s`. Here `line` and `pixel` are the spatial axes and
`channel` is the detector axis.

The fifth, `tcspc_histogram`, is the global decay: its
dimensions are `(frame, channel, tcspc_channel)` with no spatial axes.



## Roadmap

Planned updates:

- **Interactive phasor** — lasso a region of the plot and paint the matching pixels
  back into the image as a napari Labels layer
- **Decay fitting** — extract lifetimes from the TCSPC curves
- **Region-wise decay** — curves per mask and per object, not only per frame/detector
- **Wider import support** — `.ptu` from further scanning systems, and other formats
  (`.sdt`, …)



## License

Distributed under the terms of the [MIT] license.
`napari-flopa` is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[tttrkit]: https://github.com/panekdal/tttrkit
[MIT]: http://opensource.org/licenses/MIT
[file an issue]: https://github.com/cockovaz/napari-flopa/issues
[issues]: https://github.com/cockovaz/napari-flopa/issues
[pip]: https://pypi.org/project/pip/
