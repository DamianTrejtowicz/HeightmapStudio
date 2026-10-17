"""Heightmap Studio — local CPU depth estimation and native-size PNG export."""
from pathlib import Path
import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageTk, ImageDraw, ImageChops
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_propagation
from outline_relief import outline_relief, clean_support
from online_ai import ConfigStore, OnlineEngine, OnlineConfig, PROVIDERS

ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
MODEL = ROOT / 'assets' / 'depth_anything_v2_small.onnx'
VERSION = '1.5.0'
MAX_PIXELS = 300_000_000
WORK_EDGE = 2048
Image.MAX_IMAGE_PIXELS = None  # Selected local files are checked explicitly before decoding.
MODES = ['AI — shape and depth', 'Brightness — graphic relief', 'Existing heightmap — no AI', 'Illustration — outline relief']


@dataclass
class Settings:
    gamma: float = 1.0
    smooth: float = 1.0
    strength: float = 1.0
    base: float = 0.0
    detail: float = 0.0
    convex: float = 0.0
    invert: bool = False
    white_bg: bool = False
    threshold: float = 240.0
    outline_threshold: float = 70.0


def source_metadata(path):
    with Image.open(path) as f:
        size = f.size
        if f.getexif().get(274, 1) in (5, 6, 7, 8): size = size[::-1]
        if size[0] * size[1] > MAX_PIXELS:
            raise ValueError('This image exceeds 300 million pixels. Please crop it before loading.')
        return size


def read_source(path, max_edge=WORK_EDGE):
    with Image.open(path) as f:
        if f.width * f.height > MAX_PIXELS:
            raise ValueError('This image exceeds 300 million pixels. Please crop it before loading.')
        dpi = f.info.get('dpi')
        if max_edge and max(f.size) > max_edge:
            f.draft(f.mode, (max_edge, max_edge))
            f.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS,
                        reducing_gap=None if f.mode.startswith('I;16') else 2.0)
        original = ImageOps.exif_transpose(f)
        # Retain real precision of grayscale 16-bit sources.
        if original.mode in ('I;16', 'I;16L', 'I;16B', 'I', 'F'):
            gray = np.array(original, dtype=np.float32)
            if original.mode != 'F':
                gray /= 65535 if original.mode.startswith('I;16') or gray.max() > 255 else 255
            gray = np.clip(gray, 0, 1)
            im = Image.fromarray(np.uint8(gray * 255)).convert('RGBA')
        else:
            im = original.convert('RGBA')
            a = np.asarray(im, dtype=np.float32)
            gray = (a[:, :, :3] @ np.array([.299, .587, .114], np.float32)) / 255
    return im, gray, dpi


def read_document(path):
    size = source_metadata(path)
    im, gray, dpi = read_source(path)
    if not im.getchannel('A').getbbox(): raise ValueError('The image is fully transparent.')
    rgb = np.asarray(im)[:, :, :3]
    isgray = np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])
    return im, gray, dpi, size, isgray, Path(path)


class DepthEngine:
    def __init__(self):
        self.session = None

    def estimate(self, im):
        import onnxruntime as ort
        if self.session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, min(6, (os.cpu_count() or 2) - 1))
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(MODEL), sess_options=options,
                                                providers=['CPUExecutionProvider'])
        rgb = Image.new('RGBA', im.size, (128, 128, 128, 255))
        rgb.alpha_composite(im)
        rgb = rgb.convert('RGB')
        # Aspect preserving resize; cap long panoramas to bound CPU/RAM use.
        scale = min(518 / min(im.size), 1036 / max(im.size))
        tw, th = [max(14, int(round(v * scale / 14)) * 14) for v in im.size]
        pixels = np.asarray(rgb.resize((tw, th), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        pixels = (pixels - [.485, .456, .406]) / [.229, .224, .225]
        tensor = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)
        output = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0]
        depth = np.squeeze(output).astype(np.float32)
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise ValueError('The model returned an invalid depth map.')
        return np.asarray(Image.fromarray(depth).resize(im.size, Image.Resampling.BICUBIC)).copy()


def normalize_depth(raw, mask):
    valid = raw[mask > .5]
    if not valid.size:
        raise ValueError('No visible artwork remains. Adjust the background threshold or restore the mask.')
    lo, hi = np.percentile(valid, [.5, 99.5])
    if hi - lo < 1e-6:
        return np.zeros_like(raw)
    return np.clip((raw - lo) / (hi - lo), 0, 1)


def source_mask(im, settings):
    a = np.asarray(im, dtype=np.float32)
    mask = a[:, :, 3] / 255
    if settings.white_bg:
        # Only remove light pixels connected to the image border, keeping white details inside.
        low = np.min(a[:, :, :3], axis=2)
        spread = np.max(a[:, :, :3], axis=2) - low
        # Include neutral cast shadows on a white sheet, while preserving colored scales.
        bright = (low >= settings.threshold) | ((low >= settings.threshold - 55) & (spread < 16))
        seed = np.zeros_like(bright)
        seed[0] = bright[0]; seed[-1] = bright[-1]
        seed[:, 0] = bright[:, 0]; seed[:, -1] = bright[:, -1]
        background = binary_propagation(seed, mask=bright)
        mask = mask * (~background)
    return mask.astype(np.float32)


def compose(raw, im, settings, mode, gray=None, manual_mask=None):
    mask = source_mask(im, settings)
    if mode == 3:
        mask = clean_support(mask)
    if mode == 0:
        height = normalize_depth(raw, mask)
    else:
        height = np.clip(raw, 0, 1).copy()
    if settings.invert:
        height = 1 - height
    # Transparent or removed background must not bleed into the foreground during smoothing.
    sigma = settings.smooth * max(im.size) / 2048
    if sigma > .01:
        weight = gaussian_filter(mask, sigma)
        height = gaussian_filter(height * mask, sigma) / np.maximum(weight, 1e-6)
    height = np.clip(height, 0, 1) ** settings.gamma
    if settings.convex > 0:
        distance = distance_transform_edt(np.pad(mask > .5, 1))[1:-1, 1:-1]
        broad = gaussian_filter(distance, max(1, max(im.size) / 300))
        positive = broad[mask > .5]
        if positive.size and positive.max() > 0:
            # Saturating rounded silhouette: narrow regions remain gently raised.
            scale = max(float(np.percentile(positive, 85)), 1)
            volume = np.sqrt(np.clip(broad / scale, 0, 1))
            height = height * (1 - settings.convex) + volume * settings.convex
    if settings.detail > 0 and gray is not None:
        highpass = gray - gaussian_filter(gray, max(1, max(im.size) / 180))
        height = np.clip(height + settings.detail * highpass, 0, 1)
    height = (settings.base + (1 - settings.base) * height) * settings.strength
    if manual_mask is not None:
        mask *= np.asarray(manual_mask, dtype=np.float32) / 255
    return (np.clip(height, 0, 1) * mask).astype(np.float32), mask


def raster_strokes(mask, strokes, scale=1.0, offset=(0, 0)):
    """Rasterize native-pixel stroke vectors into a viewport, work mask or PNG strip."""
    painter = ImageDraw.Draw(mask)
    sx, sy = (scale,scale) if isinstance(scale,(int,float)) else scale
    for stroke in strokes:
        rx,ry = stroke['radius']*sx,stroke['radius']*sy
        points = [(p[0]*sx+offset[0], p[1]*sy+offset[1]) for p in stroke['points']]
        fill = stroke['fill']
        if len(points) > 1: painter.line(points, fill=fill, width=max(1, round(rx+ry)))
        for px, py in points:
            painter.ellipse((px-rx, py-ry, px+rx, py+ry), fill=fill)
    return mask


def export_height(path, height, mask, bits=8, alpha=False, dpi=None):
    if bits == 16:
        out = Image.fromarray(np.uint16(np.rint(height * 65535)))
    else:
        out = Image.fromarray(np.uint8(np.rint(height * 255)))
        if alpha:
            out = Image.merge('LA', (out, Image.fromarray(np.uint8(np.rint(mask * 255)))))
    kwargs = {}
    if dpi and len(dpi) == 2 and all(np.isfinite(d) and d > 0 for d in dpi):
        kwargs['dpi'] = tuple(dpi)
    path = Path(path)
    # Save beside the target then replace; a failed export never truncates an existing file.
    temp = path.with_name(path.name + '.heightmap-tmp')
    try:
        out.save(temp, format='PNG', **kwargs)
        with Image.open(temp) as check:
            check.load()
            if check.size != out.size:
                raise ValueError('Export size verification failed.')
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def export_native(path, height, mask, source_path, output_size, bits=8,
                  alpha=False, dpi=None, preserve_existing=False, progress=None, manual_strokes=None):
    """Write native-size PNG in strips, without native-size float arrays or output buffers."""
    width, rows = output_size
    maximum = 65535 if bits == 16 else 255
    dtype = np.uint16 if bits == 16 else np.uint8
    work_height = Image.fromarray(dtype(np.rint(np.clip(height, 0, 1) * maximum)))
    work_mask = Image.fromarray(np.uint8(np.rint(np.clip(mask, 0, 1) * 255)))
    path = Path(path)
    temp = path.with_name(path.name + '.heightmap-tmp')
    use_alpha = bool(alpha and bits == 8)

    def chunk(file, kind, data):
        file.write(struct.pack('>I', len(data)))
        file.write(kind); file.write(data)
        file.write(struct.pack('>I', zlib.crc32(data, zlib.crc32(kind)) & 0xffffffff))

    source = native_alpha = None
    try:
        # Decode the original only for native alpha, EXIF orientation, or an exact existing-map export.
        source = Image.open(source_path)
        orientation = source.getexif().get(274, 1)
        if orientation != 1:
            oriented = ImageOps.exif_transpose(source)
            source.close(); source = oriented
        if source.size != output_size:
            raise ValueError('Source dimensions changed after loading. Please load the image again.')
        if 'A' in source.getbands() or 'transparency' in source.info:
            if 'A' in source.getbands(): native_alpha = source.getchannel('A')
            else: native_alpha = source.convert('RGBA').getchannel('A')
        if not preserve_existing:
            source.close(); source = None
        with temp.open('wb') as file:
            file.write(b'\x89PNG\r\n\x1a\n')
            chunk(file, b'IHDR', struct.pack('>IIBBBBB', width, rows, bits, 4 if use_alpha else 0, 0, 0, 0))
            if dpi and len(dpi) == 2 and all(np.isfinite(d) and 0 < d < 1e7 for d in dpi):
                chunk(file, b'pHYs', struct.pack('>IIB', round(dpi[0] / .0254), round(dpi[1] / .0254), 1))
            compressor = zlib.compressobj(6)
            for y in range(0, rows, 128):
                count = min(128, rows-y)
                box = (0, y * work_height.height / rows, work_height.width,
                       (y+count) * work_height.height / rows)
                # Keep hard manual mask boundaries while interpolating the shape smoothly.
                m = np.array(work_mask.resize((width, count), Image.Resampling.NEAREST, box=box))
                original_alpha = None
                if native_alpha is not None:
                    original_alpha = np.asarray(native_alpha.crop((0, y, width, y+count)))
                    m = np.where(m > 0, original_alpha, 0).astype(np.uint8)
                if manual_strokes:
                    precise_mask = raster_strokes(Image.new('L', (width,count),255),manual_strokes,offset=(0,-y))
                    m[np.asarray(precise_mask)==0] = 0
                if preserve_existing:
                    tile = source.crop((0, y, width, y+count))
                    if tile.mode in ('I;16', 'I;16L', 'I;16B', 'I', 'F'):
                        a = np.asarray(tile, dtype=np.float32)
                        if tile.mode != 'F': a /= 65535 if tile.mode.startswith('I;16') or a.max() > 255 else 255
                        a = np.clip(a, 0, 1) * maximum
                    else:
                        a = np.asarray(tile.convert('L'), dtype=np.float32) * (257 if bits == 16 else 1)
                    a *= m.astype(np.float32) / 255
                    values = dtype(np.rint(a))
                else:
                    values = np.array(work_height.resize((width, count), Image.Resampling.BICUBIC, box=box))
                    values[m == 0] = 0
                if use_alpha:
                    values = np.stack((values, m), axis=-1)
                elif bits == 16:
                    values = values.astype('>u2')
                scanlines = values.reshape(count, -1).view(np.uint8).reshape(count, -1)
                filtered = np.empty((count, scanlines.shape[1]+1), dtype=np.uint8)
                filtered[:, 0] = 0; filtered[:, 1:] = scanlines
                compressed = compressor.compress(filtered.tobytes())
                if compressed: chunk(file, b'IDAT', compressed)
                if progress: progress((y+count) / rows)
            chunk(file, b'IDAT', compressor.flush())
            chunk(file, b'IEND', b'')
        with Image.open(temp) as check:
            if check.size != output_size: raise ValueError('Export size verification failed.')
            check.verify()
        os.replace(temp, path)
    finally:
        if source is not None: source.close()
        if native_alpha is not None: native_alpha.close()
        if temp.exists(): temp.unlink()


def relief_preview(height):
    gy, gx = np.gradient(height)
    nx, ny = -gx * 90, -gy * 90
    light = np.clip((-.4 * nx - .5 * ny + .72) / np.sqrt(nx * nx + ny * ny + 1), .1, 1)
    return Image.fromarray(np.uint8(light * 255))


class Application:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.engine = DepthEngine()
        self.online_engine = OnlineEngine()
        self.config_store = ConfigStore()
        self.online_config = self.config_store.load()
        self.events = queue.Queue()
        self.image = self.gray = self.raw = self.depth_image = None
        self.path = None
        self.original_size = None
        self.dpi = None
        self.busy = False
        self.preview_job = None
        self.photos = []
        self.manual_mask = None
        self.strokes = []
        self.current_stroke = None
        self.source_transform = None
        self.source_zoom = 1.0
        self.source_center = None
        self.pan_anchor = None
        self.native_source = None
        self.native_loading = False
        self.native_failed = False
        self.document_id = 0
        root.title('Heightmap Studio — UV relief heightmaps')
        root.geometry('1280x900'); root.minsize(1100, 800)
        root.configure(bg='#edf1f7')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 10), background='#edf1f7', foreground='#202c40')
        style.configure('TButton', padding=(14, 9), background='#e0e7f1')
        style.configure('Accent.TButton', background='#2464d8', foreground='white')
        style.map('Accent.TButton', background=[('active', '#174eae'), ('disabled', '#9aacc8')])
        style.configure('TCombobox', padding=5)
        style.configure('Title.TLabel', font=('Segoe UI Semibold', 23), background='#17263f', foreground='white')
        top = tk.Frame(root, bg='#17263f', padx=24, pady=16); top.pack(fill='x')
        ttk.Label(top, text='Heightmap Studio', style='Title.TLabel').pack(side='left')
        self.connection_badge = tk.Label(top, text='LOCAL AI  •  OFFLINE', bg='#17263f', fg='#9fc5ff', font=('Segoe UI', 10))
        self.connection_badge.pack(side='right')
        toolbar = ttk.Frame(root, padding=(20, 14)); toolbar.pack(fill='x')
        self.load_button = ttk.Button(toolbar, text='①  Load image', command=self.load)
        self.load_button.pack(side='left', padx=(0, 10))
        self.mode = tk.StringVar(value=MODES[0])
        self.mode_box = ttk.Combobox(toolbar, textvariable=self.mode, values=MODES, state='readonly', width=30)
        self.mode_box.pack(side='left', padx=(0, 10))
        self.mode_box.bind('<<ComboboxSelected>>', lambda e: self.mode_changed())
        self.generate_button = ttk.Button(toolbar, text='②  Generate heightmap', command=self.generate)
        self.generate_button.pack(side='left')
        self.export_button = ttk.Button(toolbar, text='③  Export PNG', style='Accent.TButton', command=self.export, state='disabled')
        self.export_button.pack(side='right')
        self.settings_button=ttk.Button(toolbar,text='Settings / accounts',command=self.open_settings)
        self.settings_button.pack(side='right',padx=10)
        services=ttk.Frame(root,padding=(20,0,20,10)); services.pack(fill='x')
        ttk.Label(services,text='AI service:').pack(side='left',padx=(0,10))
        self.provider_var=tk.StringVar(value=self.online_config.provider)
        self.provider_box=ttk.Combobox(services,textvariable=self.provider_var,values=PROVIDERS,state='readonly',width=36)
        self.provider_box.pack(side='left')
        self.provider_box.bind('<<ComboboxSelected>>',lambda e:self.provider_changed())
        self.service_hint=ttk.Label(services,text='Offline: no upload, no account needed.')
        self.service_hint.pack(side='left',padx=16)
        area = ttk.Frame(root, padding=(20, 0, 20, 0)); area.pack(fill='both', expand=True)
        area.columnconfigure(0, weight=1); area.columnconfigure(1, weight=1); area.rowconfigure(1, weight=1)
        ttk.Label(area, text='SOURCE IMAGE', font=('Segoe UI Semibold', 10)).grid(row=0, column=0, sticky='w', pady=(0, 7))
        ttk.Label(area, text='HEIGHTMAP  ·  white = highest', font=('Segoe UI Semibold', 10)).grid(row=0, column=1, sticky='w', padx=(12, 0), pady=(0, 7))
        self.source_canvas = tk.Canvas(area, bg='#dce3ed', highlightthickness=0,height=1)
        self.map_canvas = tk.Canvas(area, bg='#111722', highlightthickness=0,height=1)
        self.source_canvas.grid(row=1, column=0, sticky='nsew', padx=(0, 6))
        self.map_canvas.grid(row=1, column=1, sticky='nsew', padx=(6, 0))
        self.source_canvas.bind('<Configure>', lambda e: self.redraw_source())
        self.map_canvas.bind('<Configure>', lambda e: self.schedule_preview())
        self.source_canvas.bind('<ButtonPress-1>', self.paint_start)
        self.source_canvas.bind('<B1-Motion>', self.paint_move)
        self.source_canvas.bind('<ButtonRelease-1>', self.paint_end)
        self.source_canvas.bind('<Motion>', self.brush_cursor)
        self.source_canvas.bind('<Leave>', lambda e: self.source_canvas.delete('brush_cursor'))
        self.source_canvas.bind('<MouseWheel>', self.wheel_zoom)
        self.source_canvas.bind('<Button-4>', lambda e:self.zoom_source(1.25,e.x,e.y))
        self.source_canvas.bind('<Button-5>', lambda e:self.zoom_source(.8,e.x,e.y))
        for button in (2,3):
            self.source_canvas.bind(f'<ButtonPress-{button}>',self.pan_start)
            self.source_canvas.bind(f'<B{button}-Motion>',self.pan_move)
            self.source_canvas.bind(f'<ButtonRelease-{button}>',self.pan_end)
        zoom_bar=ttk.Frame(area)
        zoom_bar.grid(row=2,column=0,sticky='w',pady=(6,0))
        ttk.Label(zoom_bar,text='Source zoom:').pack(side='left',padx=(0,6))
        ttk.Button(zoom_bar,text='−',width=2,command=lambda:self.zoom_source(.8)).pack(side='left')
        ttk.Button(zoom_bar,text='+',width=2,command=lambda:self.zoom_source(1.25)).pack(side='left',padx=3)
        ttk.Button(zoom_bar,text='Fit',width=4,command=self.fit_source).pack(side='left',padx=3)
        ttk.Button(zoom_bar,text='100%',width=5,command=self.actual_source).pack(side='left',padx=3)
        self.zoom_label=ttk.Label(zoom_bar,text='Fit')
        self.zoom_label.pack(side='left',padx=8)
        ttk.Label(area,text='Wheel: zoom at cursor  ·  Right drag: pan',font=('Segoe UI',9)).grid(row=2,column=1,sticky='w',padx=12,pady=(6,0))
        brushes = ttk.Frame(root, padding=(20, 10, 20, 0)); brushes.pack(fill='x')
        ttk.Label(brushes, text='Manual background:').pack(side='left', padx=(0, 8))
        self.brush_mode = tk.StringVar(value='View')
        self.brush_box = ttk.Combobox(brushes, textvariable=self.brush_mode,
            values=['View', 'Erase background', 'Restore mask'], state='readonly', width=18)
        self.brush_box.pack(side='left')
        self.brush_box.bind('<<ComboboxSelected>>', lambda e: self.source_canvas.configure(cursor='crosshair' if self.brush_mode.get() != 'View' else 'hand2'))
        ttk.Label(brushes, text='Brush size:').pack(side='left', padx=(16, 6))
        self.brush_size = tk.DoubleVar(value=30)
        ttk.Scale(brushes, from_=2, to=160, variable=self.brush_size, length=130).pack(side='left')
        ttk.Button(brushes, text='Undo stroke', command=self.undo_stroke).pack(side='left', padx=(16, 6))
        ttk.Button(brushes, text='Clear manual mask', command=self.clear_mask).pack(side='left')
        ttk.Label(brushes, text='Paint on SOURCE · red = erased').pack(side='right')
        controls = ttk.Frame(root, padding=(20, 12)); controls.pack(fill='x')
        self.variables = {}
        specs = [('gamma', 'Height curve / gamma', .35, 2.5, 1), ('smooth', 'Smoothing', 0, 12, 1),
                 ('strength', 'Height range', .1, 1, 1), ('base', 'Silhouette base', 0, .5, 0),
                 ('detail', 'Fine detail', 0, .5, 0), ('convex', 'Silhouette rounding', 0, 1, 0)]
        for i, (name, title, low, high, value) in enumerate(specs):
            box = ttk.Frame(controls); box.grid(row=i // 3, column=i % 3, sticky='ew', padx=(0, 18), pady=3)
            controls.columnconfigure(i % 3, weight=1)
            label = ttk.Label(box, text=title); label.pack(anchor='w')
            var = tk.DoubleVar(value=value); self.variables[name] = var
            ttk.Scale(box, from_=low, to=high, variable=var, command=lambda v, n=name, t=title, l=label: self.slider(n, t, l)).pack(fill='x')
        opts = ttk.Frame(root, padding=(20, 0, 20, 7)); opts.pack(fill='x')
        for name, title in [('invert', 'Invert height'), ('white_bg', 'Remove light background')]:
            var = tk.BooleanVar(value=False); self.variables[name] = var
            ttk.Checkbutton(opts, text=title, variable=var, command=self.schedule_preview).pack(side='left', padx=(0, 16))
        ttk.Label(opts, text='Threshold:').pack(side='left')
        self.variables['threshold'] = tk.DoubleVar(value=240)
        ttk.Spinbox(opts, from_=180, to=255, width=5, textvariable=self.variables['threshold'], command=self.schedule_preview).pack(side='left')
        self.variables['threshold'].trace_add('write', lambda *a: self.schedule_preview())
        ttk.Label(opts, text='  Outline threshold (Generate):').pack(side='left')
        self.variables['outline_threshold'] = tk.DoubleVar(value=70)
        ttk.Spinbox(opts, from_=10, to=220, width=5, textvariable=self.variables['outline_threshold']).pack(side='left')
        self.relief = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text='Shaded preview', variable=self.relief, command=self.schedule_preview).pack(side='right')
        foot = ttk.Frame(root, padding=(20, 4, 20, 10)); foot.pack(fill='x')
        ttk.Button(foot, text='Reset settings', command=self.reset).pack(side='left')
        ttk.Label(foot, text='Export:').pack(side='left', padx=(16, 5))
        self.bits = tk.StringVar(value='PNG 8-bit')
        ttk.Combobox(foot, textvariable=self.bits, values=['PNG 8-bit', 'PNG 16-bit'], state='readonly', width=13).pack(side='left')
        self.alpha_export = tk.BooleanVar(value=False)
        ttk.Checkbutton(foot, text='Transparency (8-bit)', variable=self.alpha_export).pack(side='left', padx=12)
        ttk.Button(foot, text='Help / licenses', command=self.help).pack(side='right')
        self.progress = ttk.Progressbar(root, mode='indeterminate'); self.progress.pack(fill='x')
        self.status = tk.StringVar(value='Load PNG, JPG, WEBP, BMP or TIFF. The AI model is included in this program.')
        ttk.Label(root, textvariable=self.status, padding=(20, 9), anchor='w').pack(fill='x')
        root.after(100, self.poll)
        root.bind('<Control-o>', lambda e: self.load())
        root.bind('<Control-s>', lambda e: self.export())
        root.bind('<Control-z>', lambda e: self.undo_stroke())
        root.protocol('WM_DELETE_WINDOW', root.destroy)
        self.schedule_preview()
        self.provider_changed()
        if self.config_store.warning: self.status.set(self.config_store.warning)

    def get_settings(self):
        return Settings(**{k: v.get() for k, v in self.variables.items()})

    def slider(self, name, title, label):
        label.configure(text=f'{title}  {self.variables[name].get():.2f}')
        self.schedule_preview()

    def reset(self):
        for name, value in vars(Settings()).items(): self.variables[name].set(value)
        self.schedule_preview()

    def image_point(self, event):
        if self.source_transform is None or self.image is None: return None
        ox, oy, scale = self.source_transform
        px, py = (event.x - ox) / scale, (event.y - oy) / scale
        width,height=self.original_size or self.image.size
        if not (0 <= px < width and 0 <= py < height): return None
        return (px, py)

    def fit_scale(self):
        if self.image is None: return 1
        width,height=self.original_size or self.image.size
        return min(max(1,self.source_canvas.winfo_width()-24)/width,
                   max(1,self.source_canvas.winfo_height()-24)/height)

    def fit_source(self):
        self.source_zoom=1.0; self.source_center=None
        self.redraw_source()

    def actual_source(self):
        if self.image is None: return
        self.zoom_source(absolute=1.0)

    def zoom_source(self,factor=1.0,x=None,y=None,absolute=None):
        if self.image is None or self.current_stroke is not None: return
        cw,ch=self.source_canvas.winfo_width(),self.source_canvas.winfo_height()
        if self.source_transform is None: self.redraw_source()
        ox,oy,old_scale=self.source_transform
        x=cw/2 if x is None else x; y=ch/2 if y is None else y
        anchor=((x-ox)/old_scale,(y-oy)/old_scale)
        target=absolute if absolute is not None else old_scale*factor
        scale=max(self.fit_scale()*.25,min(16.0,target))
        self.source_zoom=scale/self.fit_scale()
        self.source_center=(anchor[0]-(x-cw/2)/scale,anchor[1]-(y-ch/2)/scale)
        self.redraw_source()

    def wheel_zoom(self,event):
        self.zoom_source(1.25**max(-4,min(4,event.delta/120)),event.x,event.y)
        return 'break'

    def pan_start(self,event):
        if self.image is None or self.current_stroke is not None: return
        self.pan_anchor=(event.x,event.y)
        self.source_canvas.configure(cursor='fleur')

    def pan_move(self,event):
        if self.pan_anchor is None or self.source_transform is None: return
        dx,dy=event.x-self.pan_anchor[0],event.y-self.pan_anchor[1]
        scale=self.source_transform[2]
        cx,cy=self.source_center
        self.source_center=(cx-dx/scale,cy-dy/scale)
        self.pan_anchor=(event.x,event.y)
        self.redraw_source()

    def pan_end(self,event):
        self.pan_anchor=None
        self.source_canvas.configure(cursor='hand2' if self.brush_mode.get()=='View' else 'crosshair')

    def redraw_source(self):
        if not hasattr(self,'zoom_label'): return
        canvas=self.source_canvas
        cw,ch=max(50,canvas.winfo_width()),max(50,canvas.winfo_height())
        canvas.delete('all')
        if self.image is None:
            self.source_transform=None
            canvas.create_text(cw//2,ch//2,text='Load an image to see a preview',fill='#8496b0',font=('Segoe UI',12))
            return
        width,height=self.original_size or self.image.size
        if self.source_center is None: self.source_center=(width/2,height/2)
        scale=self.fit_scale()*self.source_zoom
        cx,cy=self.source_center
        # Keep a small part of the image reachable even after a long drag.
        cx=min(width+cw/(2*scale)-24/scale,max(-cw/(2*scale)+24/scale,cx))
        cy=min(height+ch/(2*scale)-24/scale,max(-ch/(2*scale)+24/scale,cy))
        self.source_center=(cx,cy)
        ox,oy=cw/2-cx*scale,ch/2-cy*scale
        self.source_transform=(ox,oy,scale)
        self.zoom_label.configure(text=f'{scale*100:.0f}%'+(' · loading detail' if self.native_loading else ''))
        if (self.native_source is None and not self.native_loading and not self.native_failed and self.path is not None
                and (width,height)!=self.image.size and scale>self.image.width/width):
            self.native_loading=True
            document,path=self.document_id,self.path
            def load_detail():
                try:
                    source=Image.open(path)
                    if source.getexif().get(274,1)!=1:
                        oriented=ImageOps.exif_transpose(source); source.close(); source=oriented
                    source.load()
                    if source.size!=(width,height): source.close(); raise ValueError('Source dimensions changed. Load the image again.')
                    self.events.put(('native_source',(document,source)))
                except Exception:
                    self.events.put(('native_error',document))
            threading.Thread(target=load_detail,daemon=True).start()
        # Crop the visible region before scaling: zoom never creates a giant resized image.
        left=max(0,int(np.floor(-ox/scale))); top=max(0,int(np.floor(-oy/scale)))
        right=min(width,int(np.ceil((cw-ox)/scale))); bottom=min(height,int(np.ceil((ch-oy)/scale)))
        view=Image.new('RGBA',(cw,ch),'#dce3ed')
        if right>left and bottom>top:
            source=self.native_source if self.native_source is not None else self.image
            sx,sy=source.width/width,source.height/height
            box=(max(0,int(left*sx)),max(0,int(top*sy)),min(source.width,max(1,int(np.ceil(right*sx)))),min(source.height,max(1,int(np.ceil(bottom*sy)))))
            pw,ph=max(1,round((right-left)*scale)),max(1,round((bottom-top)*scale))
            patch=source.crop(box)
            if patch.mode in ('I;16','I;16L','I;16B','I','F'):
                levels=np.asarray(patch,dtype=np.float32)
                if patch.mode!='F': levels=levels/(65535 if patch.mode.startswith('I;16') or levels.max()>255 else 255)
                patch=Image.fromarray(np.uint8(np.clip(levels,0,1)*255))
            patch=patch.convert('RGBA').resize((pw,ph),Image.Resampling.NEAREST if scale>=2 and source.size==(width,height) else Image.Resampling.LANCZOS)
            yy,xx=np.indices((ph,pw))
            values=np.where((xx//14+yy//14)%2,218,244).astype(np.uint8)
            checker=Image.fromarray(np.stack([values,values,values],axis=-1)).convert('RGBA')
            checker.alpha_composite(patch)
            view.paste(checker,(round(ox+left*scale),round(oy+top*scale)))
        all_strokes=self.strokes+([self.current_stroke] if self.current_stroke is not None else [])
        if all_strokes:
            erased=raster_strokes(Image.new('L',(cw,ch),255),all_strokes,scale,(ox,oy))
            overlay=erased.point(lambda v:int((255-v)*.6))
            visible=Image.new('L',(cw,ch),0)
            ImageDraw.Draw(visible).rectangle((ox,oy,ox+width*scale,oy+height*scale),fill=255)
            red=Image.new('RGBA',(cw,ch),(255,40,80,0))
            red.putalpha(ImageChops.multiply(overlay,visible)); view.alpha_composite(red)
        self.source_photo=ImageTk.PhotoImage(view)
        canvas.create_image(0,0,anchor='nw',image=self.source_photo)

    def brush_cursor(self, event):
        self.source_canvas.delete('brush_cursor')
        if self.brush_mode.get() != 'View' and self.image_point(event) is not None:
            r = self.brush_size.get() / 2
            self.source_canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r,
                outline='#ff6b81' if self.brush_mode.get() == 'Erase background' else '#48d7ae',
                width=2, tags='brush_cursor')

    def apply_stroke_segment(self, p, previous=None):
        width,height=self.original_size or self.image.size
        segment=dict(self.current_stroke,points=[previous,p] if previous is not None else [p])
        raster_strokes(self.manual_mask,[segment],(self.image.width/width,self.image.height/height))

    def paint_start(self, event):
        if self.busy: return
        if self.brush_mode.get()=='View': self.pan_start(event); return
        p = self.image_point(event)
        if p is None: return
        if self.manual_mask is None: self.manual_mask = Image.new('L', self.image.size, 255)
        self.current_stroke = {'radius': self.brush_size.get()/2/self.source_transform[2],
                               'fill': 0 if self.brush_mode.get() == 'Erase background' else 255,
                               'points': [p]}
        self.apply_stroke_segment(p)
        self.redraw_source()
        self.schedule_preview()

    def paint_move(self, event):
        if self.pan_anchor is not None: self.pan_move(event); return
        self.brush_cursor(event)
        if self.busy or self.current_stroke is None: return
        p = self.image_point(event)
        if p is None: return
        previous = self.current_stroke['points'][-1]
        self.apply_stroke_segment(p, previous)
        self.current_stroke['points'].append(p)
        self.redraw_source()
        self.schedule_preview()

    def paint_end(self, event):
        if self.pan_anchor is not None: self.pan_end(event); return
        if self.current_stroke is None: return
        self.strokes.append(self.current_stroke)
        self.current_stroke = None
        self.schedule_preview()

    def undo_stroke(self):
        if self.busy or not self.strokes or self.image is None: return
        self.strokes.pop()
        self.manual_mask = Image.new('L', self.image.size, 255)
        for stroke in self.strokes:
            self.current_stroke = stroke
            previous = None
            for p in stroke['points']:
                self.apply_stroke_segment(p, previous)
                previous = p
        self.current_stroke = None
        self.schedule_preview()

    def clear_mask(self):
        if self.busy: return
        self.manual_mask = None; self.strokes = []; self.current_stroke = None
        self.schedule_preview()

    def working(self, flag, text=None):
        self.busy = flag
        for button in (self.load_button, self.generate_button,self.settings_button): button.configure(state='disabled' if flag else 'normal')
        self.mode_box.configure(state='disabled' if flag else 'readonly')
        self.provider_box.configure(state='disabled' if flag else 'readonly')
        self.export_button.configure(state='normal' if not flag and self.raw is not None else 'disabled')
        if flag: self.progress.start(12)
        else: self.progress.stop()
        if text: self.status.set(text)

    def open_settings(self):
        if self.busy: return
        from online_settings import SettingsDialog
        import copy
        config=copy.deepcopy(self.online_config)
        config.provider=self.provider_var.get()
        SettingsDialog(self.root,config,self.config_store,self.settings_saved)

    def settings_saved(self,config):
        self.online_config=config
        self.provider_var.set(config.provider)
        self.provider_changed()

    def provider_changed(self):
        online=self.provider_var.get()!=PROVIDERS[0]
        self.connection_badge.configure(text='ONLINE AI  •  UPLOAD ON GENERATE' if online else 'LOCAL AI  •  OFFLINE')
        hints={PROVIDERS[0]:'Offline: no upload, no account needed.',
               PROVIDERS[1]:'Depth estimation. Free quota / queue limits apply.',
               PROVIDERS[2]:'Experimental artistic relief. API key + provider billing.',
               PROVIDERS[3]:'Experimental artistic relief. API key + account quota / billing.'}
        self.service_hint.configure(text=hints[self.provider_var.get()])
        self.raw=None; self.working(False)
        self.status.set('Online sends a resized image to the selected provider only after you click Generate.' if online
                        else 'Offline processing selected. Load an image or click Generate heightmap.')
        self.schedule_preview()

    def mode_changed(self):
        if self.busy: return
        if self.mode.get()==MODES[0] and self.provider_var.get()!=PROVIDERS[0]:
            self.raw=None; self.working(False)
            self.status.set('Ready for online AI. Click Generate heightmap to upload the resized image.')
            self.schedule_preview()
        else: self.generate()

    def background(self, function, tag):
        def run():
            try: self.events.put((tag, function()))
            except Exception as e: self.events.put(('error', str(e)))
        threading.Thread(target=run, daemon=True).start()

    def load(self):
        from tkinter import filedialog
        if self.busy: return
        path = filedialog.askopenfilename(title='Load an image or an existing heightmap', filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff'), ('All files', '*.*')])
        if not path: return
        self.working(True, 'Loading image and preparing a memory-efficient working preview…')
        self.background(lambda: read_document(path), 'loaded')

    def generate(self):
        if self.busy or self.image is None: return
        self.active_mode = MODES.index(self.mode.get())
        self.raw = None; self.depth_image = None
        self.working(True, 'Estimating shape with local AI… The first run may take a little longer.' if self.active_mode == 0 else 'Preparing heightmap…')
        im, gray, mode = self.image, self.gray, self.active_mode
        if mode==0 and self.provider_var.get()!=PROVIDERS[0]:
            import copy
            config=copy.deepcopy(self.online_config); config.provider=self.provider_var.get()
            self.status.set('Uploading resized image to '+config.provider+'… Waiting for the selected service.')
            self.background(lambda:self.online_engine.estimate(im,config),'generated')
        else:
            settings = self.get_settings()
            self.background(lambda: self.engine.estimate(im) if mode == 0 else
                            outline_relief(im, source_mask(im, settings), threshold=settings.outline_threshold)[0] if mode == 3 else gray.copy(), 'generated')

    def poll(self):
        from tkinter import messagebox
        try:
            while True:
                tag, value = self.events.get_nowait()
                if tag == 'loaded':
                    if self.native_source is not None: self.native_source.close()
                    self.native_source=None; self.native_loading=False; self.native_failed=False
                    self.document_id+=1
                    self.source_zoom=1.0; self.source_center=None; self.pan_anchor=None
                    self.image, self.gray, self.dpi, self.original_size, isgray, self.path = value
                    self.manual_mask = None; self.strokes = []; self.current_stroke = None
                    self.raw = None; self.depth_image = None; self.source_transform = None
                    self.reset()
                    self.mode.set(MODES[2] if isgray else MODES[0])
                    if isgray: self.variables['smooth'].set(0)
                    self.working(False)
                    if not isgray and self.provider_var.get()!=PROVIDERS[0]:
                        self.status.set('Image loaded. Click Generate heightmap to send a resized copy to the selected online service.')
                        self.schedule_preview()
                    else: self.generate()
                elif tag == 'generated':
                    self.raw = value; self.working(False)
                    ow, oh = self.original_size or self.image.size
                    self.status.set(f'{self.path.name}  •  export {ow} × {oh} px  •  working {self.image.width} × {self.image.height} px  •  heightmap ready')
                    self.schedule_preview()
                elif tag == 'exported':
                    self.working(False, f'Saved and verified: {value}')
                    messagebox.showinfo('Export complete', f'Heightmap saved:\n{value}')
                elif tag == 'error':
                    self.working(False, 'Operation failed. Adjust the settings or load another image.')
                    messagebox.showerror('Error', value)
                elif tag == 'export_progress':
                    self.status.set(f'Exporting at original dimensions… {value}%')
                elif tag == 'native_source':
                    document,source=value
                    if document==self.document_id:
                        self.native_source=source; self.native_loading=False
                        self.redraw_source()
                    else: source.close()
                elif tag == 'native_error' and value==self.document_id:
                    self.native_loading=False; self.native_failed=True
                    self.status.set('Original detail could not be loaded. Working preview remains available; check the source file.')
                    self.redraw_source()
        except queue.Empty: pass
        self.root.after(100, self.poll)

    def schedule_preview(self):
        if self.preview_job: self.root.after_cancel(self.preview_job)
        self.preview_job = self.root.after(140, self.render)

    def draw(self, canvas, im, checker=False):
        cw, ch = max(50, canvas.winfo_width()), max(50, canvas.winfo_height())
        canvas.delete('all')
        if im is None:
            canvas.create_text(cw // 2, ch // 2, text='Load an image to see a preview', fill='#8496b0', font=('Segoe UI', 12)); return
        thumb = im.copy(); thumb.thumbnail((cw - 24, ch - 24), Image.Resampling.LANCZOS)
        if checker:
            yy, xx = np.indices((thumb.height, thumb.width))
            v = np.where((xx // 14 + yy // 14) % 2, 218, 244).astype(np.uint8)
            back = Image.fromarray(np.stack([v, v, v], axis=-1)).convert('RGBA')
            back.alpha_composite(thumb.convert('RGBA')); thumb = back
        photo = ImageTk.PhotoImage(thumb)
        self.photos.append(photo)
        canvas.create_image(cw // 2, ch // 2, image=photo)

    def render(self):
        self.preview_job = None; self.photos = []
        self.redraw_source()
        if self.raw is None:
            self.draw(self.map_canvas, None); return
        try:
            settings = self.get_settings()
            small = self.image.copy()
            if self.active_mode != 3:
                small.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            raw = np.asarray(Image.fromarray(self.raw).resize(small.size, Image.Resampling.BICUBIC))
            gray = np.asarray(Image.fromarray(self.gray).resize(small.size, Image.Resampling.BICUBIC))
            manual = self.manual_mask.resize(small.size, Image.Resampling.NEAREST) if self.manual_mask is not None else None
            height, _ = compose(raw, small, settings, self.active_mode, gray, manual)
            preview = relief_preview(height) if self.relief.get() else Image.fromarray(np.uint8(np.rint(height * 255)))
            self.draw(self.map_canvas, preview)
        except Exception as e:
            self.map_canvas.delete('all')
            self.map_canvas.create_text(20, 30, anchor='nw', text=str(e), fill='#f7b2ab', width=450)

    def export(self):
        from tkinter import filedialog, messagebox
        if self.busy or self.raw is None: return
        bits = 16 if '16' in self.bits.get() else 8
        if bits == 16 and self.alpha_export.get():
            messagebox.showinfo('16-bit export', '16-bit PNG will be saved without alpha. Removed background will have zero height.');
        path = filedialog.asksaveasfilename(title='Export PNG heightmap', defaultextension='.png', initialfile=self.path.stem + f'_height_{bits}bit.png', filetypes=[('PNG', '*.png')])
        if not path: return
        if Path(path).resolve() == self.path.resolve():
            messagebox.showerror('Choose another filename', 'Export cannot overwrite the source image.'); return
        try: settings = self.get_settings()
        except Exception:
            messagebox.showerror('Invalid settings', 'Check the background threshold value.'); return
        raw, im, gray, mode, dpi = self.raw, self.image, self.gray, self.active_mode, self.dpi
        output_size = self.original_size or im.size
        source_path = self.path
        manual = self.manual_mask.copy() if self.manual_mask is not None else None
        import copy
        strokes=copy.deepcopy(self.strokes)
        alpha = self.alpha_export.get()
        self.working(True, 'Exporting at full source resolution and verifying the saved file…')
        def save():
            height, mask = compose(raw, im, settings, mode, gray, manual if output_size==im.size else None)
            preserve = (mode == 2 and settings.gamma == 1 and settings.smooth == 0
                        and settings.strength == 1 and settings.base == 0
                        and settings.detail == 0 and settings.convex == 0 and not settings.invert)
            if output_size == im.size and not preserve:
                export_height(path, height, mask, bits, alpha, dpi)
            else:
                last = [-1]
                def report(fraction):
                    percent = int(fraction * 100)
                    if percent != last[0]:
                        last[0] = percent
                        self.events.put(('export_progress', percent))
                export_native(path, height, mask, source_path, output_size, bits, alpha, dpi, preserve, report,
                              manual_strokes=strokes if output_size!=im.size else None)
            return path
        self.background(save, 'exported')

    def help(self):
        from tkinter import messagebox
        messagebox.showinfo('Heightmap Studio — help',
            '1. Load an image — a heightmap is generated automatically.\n'
            '2. Adjust shape, smoothing and detail; white = highest.\n'
            'Illustration mode inflates closed areas between dark outlines.\n'
            'For opaque backgrounds: remove background, then click Generate again.\n'
            '3. Export PNG. Original dimensions and proportions are preserved.\n\n'
            'Choose AI service or open Settings / accounts to enter your own API keys.\n'
            'Online uploads happen only when Generate is clicked; free services have quotas.\n'
            'OpenAI / Gemini are experimental artistic reliefs and can alter geometry.\n'
            'Keys are session-only unless you enable encrypted storage in Settings.\n\n'
            'Large images use a working preview up to 2048 px. Export retains original dimensions.\n'
            'Existing heightmaps with default settings preserve native pixel detail on export.\n\n'
            'Manual background: select Erase background, then paint on SOURCE.\n'
            'Red marks show removed areas. Use Restore mask, Undo stroke or Ctrl+Z.\n'
            'Wheel / + / - zooms the source at the cursor. Right drag pans in any brush mode.\n'
            'Fit shows the whole source. 100% shows one original pixel per screen pixel.\n'
            'Brush size stays constant on screen; zoom in for smaller, precise strokes.\n'
            'Restore only reverses manual erasing; turn off automatic background removal if needed.\n\n'
            'AI: local Depth Anything V2 Small, Apache 2.0.\n'
            'Brightness maps tones to height. Existing heightmap preserves input levels.\n'
            'Remove light background removes light areas and neutral shadows connected to the border.\n'
            'Silhouette rounding works best on a masked or transparent background.\n'
            'Shading is preview-only. Export always contains height data.\n\n'
            'AI estimates relative shape, not physical height in mm. '
            'Illustrations may need adjustments and a test print. '
            'Set physical height in xTool Studio. Use PNG 8-bit for basic compatibility; '
            'custom heightmaps and 16-bit PNG support depend on the importing software.\n\n'
            'Model ONNX: onnx-community/depth-anything-v2-small.\n'
            'Python, Tk, Pillow, NumPy, SciPy, ONNX Runtime, PyInstaller. '
            'Model and dependency licenses are bundled inside the EXE.\n'
            f'Version {VERSION}. Offline does not upload. Online sends only a resized working copy.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', nargs=2, metavar=('SOURCE', 'OUTPUT_DIR'))
    parser.add_argument('--outline-test', nargs=2, metavar=('SOURCE', 'OUTPUT_DIR'))
    parser.add_argument('--ui-test', nargs=2, metavar=('SOURCE', 'SCREENSHOT'))
    parser.add_argument('--online-test', metavar='OUTPUT_DIR')
    parser.add_argument('--settings-test', metavar='SCREENSHOT')
    parser.add_argument('--zoom-test',nargs=2,metavar=('SOURCE','SCREENSHOT'))
    args = parser.parse_args()
    if args.outline_test:
        source, destination = args.outline_test
        destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
        im, gray, dpi = read_source(source)
        raw, mask, info = outline_relief(im)
        height, mask = compose(raw, im, Settings(smooth=0), 3, gray)
        size = source_metadata(source)
        for bits in (8, 16):
            export_native(destination / f'outline_{bits}bit.png', height, mask, source, size, bits, dpi=dpi)
        export_native(destination / 'outline_alpha.png', height, mask, source, size, 8, True, dpi)
        info.update(version=VERSION, size=size, working_size=im.size,
                    zero_background=bool(np.all(height[mask == 0] == 0)))
        (destination / 'outline_test.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        return
    if args.online_test:
        destination=Path(args.online_test); destination.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        im=Image.new('RGBA',(384,256),(255,255,255,255))
        draw=ImageDraw.Draw(im)
        draw.ellipse((80,25,300,230),fill=(60,140,210,255))
        draw.ellipse((120,60,245,190),fill=(230,170,80,255))
        config=OnlineConfig(provider=PROVIDERS[1],timeout=150)
        raw=OnlineEngine().estimate(im,config)
        mask=np.ones(raw.shape,dtype=np.float32)
        height=normalize_depth(raw,mask)
        export_height(destination/'online_depth.png',height,mask)
        report={'version':VERSION,'provider':config.provider,'space':config.hf_space,
                'anonymous_live_request':True,'size':list(im.size),'finite':bool(np.isfinite(raw).all()),
                'seconds':round(time.monotonic()-started,2)}
        (destination/'online_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return
    if args.self_test:
        source, destination = args.self_test
        destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        im, gray, dpi = read_source(source)
        native_size = source_metadata(source)
        raw = DepthEngine().estimate(im)
        settings = Settings(white_bg=True, threshold=235, convex=.25, detail=.12)
        height, mask = compose(raw, im, settings, 0, gray)
        for bits in (8, 16):
            export_native(destination / f'test_height_{bits}bit.png', height, mask, source, native_size, bits, dpi=dpi)
        export_native(destination / 'test_height_alpha.png', height, mask, source, native_size, 8, True, dpi)
        report = {'version': VERSION, 'size': native_size, 'working_size': im.size, 'seconds': round(time.monotonic()-started, 2),
                  'model_loaded': True, 'range': [float(height.min()), float(height.max())],
                  'zero_background': bool(np.all(height[mask == 0] == 0)), 'finite': bool(np.isfinite(height).all())}
        (destination / 'self_test.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return
    import tkinter as tk
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception: pass
    root = tk.Tk()
    gui = Application(root)
    if args.settings_test:
        from online_settings import SettingsDialog
        dialog=SettingsDialog(root,OnlineConfig(provider=PROVIDERS[1]),gui.config_store,gui.settings_saved)
        def capture_settings():
            from PIL import ImageGrab
            import ctypes
            dialog.window.update_idletasks()
            hwnd=ctypes.windll.user32.GetParent(dialog.window.winfo_id())
            ImageGrab.grab(window=hwnd).save(args.settings_test)
            root.destroy()
        root.after(1000,capture_settings)
    if args.ui_test or args.zoom_test:
        source, screenshot = args.ui_test or args.zoom_test
        gui.image, gui.gray, gui.dpi = read_source(source)
        gui.original_size = source_metadata(source)
        gui.path = Path(source)
        gui.active_mode = 0
        gui.raw = gui.gray.copy() if args.zoom_test else gui.engine.estimate(gui.image)
        gui.variables['white_bg'].set(True)
        gui.variables['threshold'].set(235)
        gui.variables['convex'].set(.25)
        gui.variables['detail'].set(.12)
        gui.working(False, 'AI heightmap ready • local processing • full source resolution export')
        def capture():
            from PIL import ImageGrab
            import ctypes
            if args.zoom_test:
                if gui.native_loading:
                    root.after(100,capture); return
                assert gui.native_source is not None or gui.original_size==gui.image.size
                width,height=gui.original_size
                gui.source_center=(width/2,height/2)
                gui.zoom_source(absolute=4)
                point=(width//2,height//2)
                ox,oy,scale=gui.source_transform
                from types import SimpleNamespace
                event=SimpleNamespace(x=ox+point[0]*scale,y=oy+point[1]*scale)
                gui.brush_mode.set('Erase background'); gui.brush_size.set(8)
                gui.paint_start(event); gui.paint_end(event)
                mask=np.ones_like(gui.gray)
                target=Path(screenshot).with_suffix('.mask.png')
                export_native(target,gui.gray,mask,source,gui.original_size,8,preserve_existing=True,manual_strokes=gui.strokes)
                with Image.open(target) as image:
                    assert image.size==gui.original_size
                    assert image.getpixel(point)==0
                    assert image.getpixel((point[0]+3,point[1]))>0
                report={'version':VERSION,'original_size':gui.original_size,'native_detail':gui.native_source is not None,
                        'zoom_scale':scale,'native_brush_radius':gui.strokes[0]['radius'],
                        'native_export_erased_center_retained_neighbor':True}
                Path(screenshot).with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            else: gui.zoom_source(2)
            gui.render(); root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            ImageGrab.grab(window=hwnd).save(screenshot)
            root.destroy()
        if args.zoom_test:
            root.update_idletasks(); gui.actual_source()
        root.after(1000, capture)
    root.mainloop()


if __name__ == '__main__':
    try: main()
    except Exception:
        log = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'HeightmapStudio'
        log.mkdir(parents=True, exist_ok=True)
        (log / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        if any(flag in sys.argv for flag in ('--self-test','--outline-test','--online-test','--ui-test','--settings-test','--zoom-test')):
            sys.exit(1)
        import tkinter.messagebox
        tkinter.messagebox.showerror('Heightmap Studio', f'The application could not start. Details:\n{log / "error.log"}')
