# Heightmap Studio

A Windows desktop tool for creating grayscale heightmaps from images, with offline AI, side-by-side previews and precise manual background erasing. Written in Python with an English interface and distributed as a single executable.

**[Download the latest Windows EXE](https://github.com/DamianTrejtowicz/HeightmapStudio/releases/latest)** · [Full user guide](USER_GUIDE.txt)

## Features

- Offline CPU depth estimation using bundled Depth Anything V2 Small.
- AI depth, brightness relief and existing-heightmap modes.
- Source and heightmap previews; gamma, smoothing, height range, silhouette and detail controls.
- Automatic light-background removal and manual erase/restore brushes with undo.
- Cursor-centered zoom, Fit, 100% native-pixel view and drag-to-pan for accurate masking.
- Native-resolution PNG export in 8-bit or 16-bit grayscale; optional transparency for 8-bit output.
- Original dimensions, proportions and available DPI metadata are preserved.
- Large-image loading and export in background tasks; A3 at 600 DPI was tested.
- Optional Hugging Face, OpenAI and Gemini services with separate account settings.

## Run the application

Download `HeightmapStudio_v1.4.exe` from [Releases](https://github.com/DamianTrejtowicz/HeightmapStudio/releases), then open it on Windows x64. Python and the offline model are included. The executable extracts its bundled components into a temporary folder at startup. It is not code-signed.

1. Click **Load image** and choose an image.
2. Choose a processing mode and compare the previews. White means highest; black means zero height.
3. For manual cleanup, select **Erase background** and paint on the source. Use the mouse wheel to zoom and the right or middle button to pan. **Restore mask**, **Undo stroke** and **Clear manual mask** correct mistakes.
4. Adjust the height controls and click **Export PNG**.

The original image is never modified. Manual strokes are stored in original-pixel coordinates and applied at full export resolution. The smaller heightmap preview may not display the finest erased details.

## Online services and privacy

Offline mode requires no account or internet. Online AI sends a resized working image **only when you click Generate heightmap**. Loading an image, selecting a service or opening settings does not upload it.

- **Hugging Face:** public Depth Anything V2 Space, with anonymous quota and queue limits; an optional token can be entered in Settings. Availability and free quotas are controlled by the service.
- **OpenAI / Gemini:** experimental artistic relief generation with your own API key. These services can change outlines and details and may incur provider charges. They are not equivalent to geometric depth estimation.

Keys stay in memory by default. **Remember keys on this PC** saves them using Windows DPAPI for the current Windows user. Settings are stored outside the repository under `%LOCALAPPDATA%\HeightmapStudio`. API keys are not bundled with the application. Chat subscriptions do not replace provider API credentials.

## Use with UV printing

Heightmaps encode relative height, not millimeters. Set physical relief height in the printer software. This project is independent of xTool; custom heightmap import and 16-bit support depend on the receiving software and its version. Two imported images are not automatically linked as color and depth. Inspect alignment and make a small test print.

Single-image depth is approximate. AI inference uses a reduced image; a larger export preserves canvas size but does not add geometric detail. Sources up to 300 million pixels are accepted, subject to available RAM. Zooming into large originals requires memory to decode them.

## Run from source / build a single EXE

Use **Windows x64 with Python 3.12** and Tkinter. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe build.py --model-only
.\.venv\Scripts\python.exe heightmap_studio.py
```

Build the standalone executable:

```powershell
.\.venv\Scripts\python.exe build.py
```

Output: `dist/HeightmapStudio.exe`. The build script downloads a pinned ~99 MB model and verifies its SHA-256 before use. Model weights and generated executables are excluded from Git. Downloading dependencies and the model requires internet; the resulting offline application does not.

## Validation

Run from the project folder on Windows with a desktop session:

```powershell
.\.venv\Scripts\python.exe test_core.py
.\.venv\Scripts\python.exe test_zoom.py
.\.venv\Scripts\python.exe test_online.py
```

These cover PNG precision, transparency, background removal, native-pixel masking, zoom/pan/undo, encrypted settings, and mocked online request/response contracts. Generated fixtures and reports go into ignored `.test-output/` folders. The online regression suite does not make paid API calls.

The v1.4 executable was also checked with local model inference and a 7016 × 9921 px A3 image, including a one-pixel erase at 400% zoom and full-size export. A live anonymous Hugging Face request passed during v1.3 validation. OpenAI and Gemini were tested with simulated responses; live paid account access was not tested.

## Model and dependencies

Depth Anything V2 Small is distributed under Apache-2.0. See [third-party notices](THIRD_PARTY_NOTICES.md), [model license](assets/MODEL_LICENSE.txt), [model card](assets/MODEL_CARD.md) and [pinned provenance](assets/MODEL_PROVENANCE.json). Third-party licenses apply to their respective components; the model license does not license this application's source code.

No application source license has been selected for this repository. Public availability alone does not grant an open-source license.
