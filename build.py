"""Download the verified model and build a standalone Windows executable."""
from pathlib import Path
import hashlib
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL_URL = ('https://huggingface.co/onnx-community/depth-anything-v2-small/'
             'resolve/4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx/model.onnx')
MODEL_SHA256 = 'afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c'


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_model():
    assets = ROOT / 'assets'
    assets.mkdir(exist_ok=True)
    model = assets / 'depth_anything_v2_small.onnx'
    if model.exists():
        if checksum(model) != MODEL_SHA256:
            raise RuntimeError('Model checksum mismatch. Remove the model file and retry.')
        return model
    temporary = model.with_suffix('.onnx.download')
    try:
        print('Downloading Depth Anything V2 Small (~99 MB)...')
        urllib.request.urlretrieve(MODEL_URL, temporary)
        if checksum(temporary) != MODEL_SHA256:
            raise RuntimeError('Downloaded model checksum mismatch; build stopped.')
        temporary.replace(model)
    finally:
        temporary.unlink(missing_ok=True)
    return model


def main():
    if '--model-only' not in sys.argv and sys.platform != 'win32':
        raise SystemExit('Build the Windows executable on Windows.')
    model = ensure_model()
    print('Verified model SHA256:', checksum(model))
    if '--model-only' in sys.argv:
        return
    subprocess.run([
        sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onefile', '--windowed',
        '--name', 'HeightmapStudio', '--distpath', str(ROOT / 'dist'),
        '--workpath', str(ROOT / 'build'), '--specpath', str(ROOT),
        '--add-data', f'{ROOT / "assets"};assets', '--collect-all', 'onnxruntime',
        '--copy-metadata', 'numpy', '--copy-metadata', 'scipy',
        '--copy-metadata', 'pillow', '--copy-metadata', 'onnxruntime',
        '--collect-all', 'gradio_client', '--copy-metadata', 'huggingface_hub',
        '--copy-metadata', 'requests', '--copy-metadata', 'certifi',
        str(ROOT / 'heightmap_studio.py'),
    ], check=True)


if __name__ == '__main__':
    main()
