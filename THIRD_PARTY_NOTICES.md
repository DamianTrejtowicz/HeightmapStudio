# Third-party notices

## Bundled depth model

- Upstream: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).
- Model: Depth Anything V2 **Small**, Apache License 2.0.
- ONNX conversion: [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small).
- Pinned revision and checksum: [MODEL_PROVENANCE.json](assets/MODEL_PROVENANCE.json).
- Original license: [MODEL_LICENSE.txt](assets/MODEL_LICENSE.txt).
- Original conversion model card: [MODEL_CARD.md](assets/MODEL_CARD.md).

The application consumes the ONNX weights without modifying them. This repository does not contain weights; the build script downloads and verifies them. The Windows executable bundles the model and these notices.

## Runtime and packaging

The application uses Python/Tk, NumPy, Pillow, SciPy, ONNX Runtime, Requests, Gradio Client and Hugging Face Hub, plus their dependencies. PyInstaller packages the Windows executable. Dependency versions are recorded in `requirements.txt`; their license metadata is included in the executable's bundled distributions where provided by those packages. Each dependency remains governed by its own license. PyInstaller's bootloader exception permits distribution of bundled applications under their own terms.

Optional hosted services are provided by their respective operators and are subject to their terms, quotas and model licenses. No ownership of third-party models or services is claimed.
