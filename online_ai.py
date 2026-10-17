"""Optional online depth services. No credentials are read from environment variables."""
from dataclasses import dataclass, field
from pathlib import Path
import base64
import copy
import ctypes
from ctypes import wintypes
import io
import json
import os
import re
import tempfile
import numpy as np
from PIL import Image

PROVIDERS = ['Offline — local AI', 'Hugging Face — free quota / token',
             'OpenAI — image API', 'Google Gemini — image API']
DEFAULT_SPACE = 'depth-anything/Depth-Anything-V2'
PROMPT = (
    'Convert the supplied image to a technical grayscale bas-relief HEIGHTMAP for UV printing. '
    'White means highest, black means zero height. Infer smoothly rounded 3D volume for objects, '
    'with shallow fine surface detail. Remove cast shadows and visual lighting from the height data. '
    'Keep every object, outline, pose, scale and pixel position precisely aligned with the input. '
    'Do not redraw, invent details, move objects, crop, rotate or add text. '
    'The canvas is a square with possible black padding. Keep padding and empty background black. '
    'Return one square grayscale depth map only, without color, shading, borders or a side-by-side view.'
)


@dataclass
class OnlineConfig:
    provider: str = PROVIDERS[0]
    hf_space: str = DEFAULT_SPACE
    openai_model: str = 'gpt-image-2.5-sunburst'
    gemini_model: str = 'gemini-nano-banana-2.1'
    timeout: int = 240
    remember: bool = False
    keys: dict = field(default_factory=dict, repr=False)


def dpapi(data, decrypt=False):
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(data)
    inp = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    out = Blob()
    crypt = ctypes.WinDLL('crypt32', use_last_error=True)
    if decrypt:
        ok = crypt.CryptUnprotectData(ctypes.byref(inp), None, None, None, None, 1, ctypes.byref(out))
    else:
        ok = crypt.CryptProtectData(ctypes.byref(inp), 'Heightmap Studio', None, None, None, 1, ctypes.byref(out))
    if not ok: raise OSError('Windows could not protect/unlock the saved API credentials.')
    try: return ctypes.string_at(out.data, out.size)
    finally:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree(ctypes.cast(out.data, ctypes.c_void_p))


class ConfigStore:
    def __init__(self, folder=None):
        self.folder = Path(folder) if folder is not None else Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'HeightmapStudio'
        self.path = self.folder / 'settings.json'
        self.warning = ''

    def load(self):
        config = OnlineConfig()
        if not self.path.exists(): return config
        try:
            if self.path.stat().st_size > 200_000: raise ValueError('Oversized settings file')
            data = json.loads(self.path.read_text(encoding='utf-8'))
            for name in ('provider', 'hf_space', 'openai_model', 'gemini_model', 'timeout', 'remember'):
                if name in data: setattr(config, name, data[name])
            if config.provider not in PROVIDERS: config.provider = PROVIDERS[0]
            if config.remember and data.get('protected_keys'):
                config.keys = json.loads(dpapi(base64.b64decode(data['protected_keys']), True).decode('utf-8'))
            validate(config)
        except Exception:
            self.warning = 'Saved settings could not be read. Open Settings and enter the configuration again.'
            return OnlineConfig()
        return config

    def save(self, config):
        validate(config)
        data = {k: v for k, v in vars(config).items() if k != 'keys'}
        if config.remember and any(config.keys.values()):
            encrypted = dpapi(json.dumps(config.keys).encode('utf-8'))
            data['protected_keys'] = base64.b64encode(encrypted).decode('ascii')
        self.folder.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
        os.replace(temp, self.path)


def validate(config):
    if config.provider not in PROVIDERS: raise ValueError('Select a supported AI service.')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', config.hf_space):
        raise ValueError('Hugging Face Space must use the form owner/space-name.')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', config.openai_model): raise ValueError('Invalid OpenAI model name.')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', config.gemini_model): raise ValueError('Invalid Gemini model name.')
    if not 30 <= int(config.timeout) <= 600: raise ValueError('Timeout must be between 30 and 600 seconds.')
    if any(any(char.isspace() for char in key) for key in config.keys.values()):
        raise ValueError('API keys must not contain spaces or line breaks.')


def safe_message(exc, keys=()):
    message = str(exc)
    for key in keys:
        if key: message = message.replace(key, '[hidden]')
    message = re.sub(r'(?i)(bearer\s+)[^\s\"\']+', r'\1[hidden]', message)
    return message[:700]


def request_json(method, url, timeout, headers=None, **kwargs):
    import requests
    response = requests.request(method, url, headers=headers or {}, timeout=(15, timeout), **kwargs)
    try:
        data = response.json()
    except ValueError:
        if response.status_code >= 400: raise RuntimeError(f'Service returned HTTP {response.status_code}. Try again later.')
        raise RuntimeError('Service returned an unreadable response.')
    if response.status_code >= 400:
        error = data.get('error', {})
        message = error.get('message', '') if isinstance(error, dict) else str(error)
        hints = {401: 'Check the API key in Settings.', 403: 'Account/model access was denied.',
                 404: 'The selected model or endpoint is unavailable.',
                 429: 'Quota or rate limit reached. Wait, change service or use Offline.'}
        hint = hints.get(response.status_code, 'The service could not finish the request.')
        raise RuntimeError(f'HTTP {response.status_code}: {hint}\n{message[:350]}')
    return data


def encode_square(im):
    resized = im.copy(); resized.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    canvas = Image.new('RGBA', (1024, 1024), (0, 0, 0, 255))
    x, y = (1024-resized.width)//2, (1024-resized.height)//2
    canvas.alpha_composite(resized, (x, y))
    buffer = io.BytesIO(); canvas.convert('RGB').save(buffer, format='PNG')
    return buffer.getvalue(), (x, y, x+resized.width, y+resized.height)


def decode_generated(encoded, size, crop):
    image_bytes = base64.b64decode(encoded, validate=True)
    if len(image_bytes) > 64_000_000: raise ValueError('The returned image is too large.')
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.width * image.height > 24_000_000: raise ValueError('The returned image dimensions are too large.')
        if abs(image.width/image.height-1) > .03:
            raise ValueError('The model changed the canvas proportions. Retry or use a depth-estimation service.')
        box = tuple(round(v * image.width / 1024) if i % 2 == 0 else round(v * image.height / 1024)
                    for i, v in enumerate(crop))
        result = image.crop(box).convert('L').resize(size, Image.Resampling.BICUBIC)
        return np.asarray(result, dtype=np.float32).copy()


def gemini_image(data):
    # Current Interactions API, plus older response forms for account/model compatibility.
    for step in data.get('steps', []):
        if step.get('type') == 'model_output':
            for item in step.get('content', []):
                if item.get('type') == 'image' and item.get('data'): return item['data']
    for item in data.get('outputs', []):
        if item.get('type') == 'image' and item.get('data'): return item['data']
    raise RuntimeError('Gemini returned no image. Check model image-generation access or retry.')


class OnlineEngine:
    def estimate(self, im, config):
        config = copy.deepcopy(config)
        validate(config)
        try:
            if config.provider == PROVIDERS[1]: return self.huggingface(im, config)
            encoded, crop = encode_square(im)
            if config.provider == PROVIDERS[2]:
                key = config.keys.get('openai', '').strip()
                if not key: raise ValueError('Enter your OpenAI API key in Settings first.')
                data = request_json('POST', 'https://api.openai.com/v1/images/edits', int(config.timeout),
                    headers={'Authorization': 'Bearer ' + key},
                    data={'model': config.openai_model, 'prompt': PROMPT, 'size': '1024x1024', 'quality': 'medium', 'n': '1'},
                    files={'image[]': ('source.png', encoded, 'image/png')})
                images = data.get('data', [])
                if not images or not images[0].get('b64_json'): raise RuntimeError('OpenAI returned no PNG image.')
                return decode_generated(images[0]['b64_json'], im.size, crop)
            if config.provider == PROVIDERS[3]:
                key = config.keys.get('gemini', '').strip()
                if not key: raise ValueError('Enter your Gemini API key in Settings first.')
                data = request_json('POST', 'https://generativelanguage.googleapis.com/v1beta/interactions', int(config.timeout),
                    headers={'x-goog-api-key': key},
                    json={'model': config.gemini_model, 'store': False,
                          'input': [{'type': 'text', 'text': PROMPT},
                                    {'type': 'image', 'data': base64.b64encode(encoded).decode('ascii'), 'mime_type': 'image/png'}],
                          'response_format': {'type': 'image', 'aspect_ratio': '1:1', 'image_size': '1K', 'mime_type': 'image/png'}})
                return decode_generated(gemini_image(data), im.size, crop)
            raise ValueError('Select an online service in the AI service list.')
        except Exception as exc:
            message = safe_message(exc, config.keys.values())
            raise RuntimeError(message) from None

    def huggingface(self, im, config):
        from gradio_client import Client, handle_file
        client = job = None
        with tempfile.TemporaryDirectory(prefix='heightmap-online-') as temp:
            small = im.copy(); small.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            source = Image.new('RGBA', small.size, (128,128,128,255)); source.alpha_composite(small)
            path = Path(temp)/'source.png'; source.convert('RGB').save(path)
            try:
                client = Client(config.hf_space, token=config.keys.get('hf') or False, verbose=False,
                                analytics_enabled=False, download_files=temp,
                                httpx_kwargs={'timeout': 30})
                job = client.submit(handle_file(str(path)), api_name='/on_submit')
                result = job.result(timeout=int(config.timeout))
                if not isinstance(result, (list, tuple)) or len(result) < 2:
                    raise RuntimeError('This Space does not return the expected depth-map files.')
                gray_path = result[1]
                if isinstance(gray_path, dict): gray_path = gray_path.get('path')
                with Image.open(gray_path) as image:
                    if image.width*image.height > 24_000_000: raise ValueError('Returned map is too large.')
                    if abs((image.width/image.height)/(small.width/small.height)-1)>.03:
                        raise ValueError('Returned depth map does not match source proportions.')
                    return np.asarray(image.convert('L').resize(im.size,Image.Resampling.BICUBIC), dtype=np.float32).copy()
            finally:
                if job is not None and not job.done(): job.cancel()
                if client is not None: client.close()

    def check(self, config):
        validate(config)
        try:
            if config.provider == PROVIDERS[0]: return 'Offline mode is available. No account is required.'
            if config.provider == PROVIDERS[1]:
                key = config.keys.get('hf', '')
                headers = {'Authorization': 'Bearer '+key} if key else {}
                if key: request_json('GET', 'https://huggingface.co/api/whoami-v2', 30, headers=headers)
                data = request_json('GET','https://huggingface.co/api/spaces/'+config.hf_space,30,headers=headers)
                stage=data.get('runtime',{}).get('stage','unknown')
                return f'Space found. Runtime: {stage}. Free quota and queue availability apply.'
            if config.provider == PROVIDERS[2]:
                key=config.keys.get('openai','').strip()
                if not key: raise ValueError('Enter your OpenAI API key first.')
                request_json('GET','https://api.openai.com/v1/models/'+config.openai_model,30,
                             headers={'Authorization':'Bearer '+key})
                return 'API key and model metadata are accessible. Image-generation billing/access may still apply.'
            key=config.keys.get('gemini','').strip()
            if not key: raise ValueError('Enter your Gemini API key first.')
            request_json('GET','https://generativelanguage.googleapis.com/v1beta/models/'+config.gemini_model,30,
                         headers={'x-goog-api-key':key})
            return 'API key and model metadata are accessible. Image-generation billing/access may still apply.'
        except Exception as exc: raise RuntimeError(safe_message(exc,config.keys.values())) from None
