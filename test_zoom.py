from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import json
import tempfile
import time
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageGrab
import heightmap_studio as app
from online_ai import ConfigStore

folder=Path(__file__).resolve().parent/'.test-output'/'zoom'
folder.mkdir(parents=True,exist_ok=True)
native_size=(4096,3072)
path=folder/'native_detail.png'
source=Image.new('RGB',native_size,(170,180,190))
draw=ImageDraw.Draw(source)
draw.line((2051,0,2051,3071),fill=(5,20,250),width=1)
draw.ellipse((2030,1510,2070,1550),outline=(250,40,70),width=1)
source.save(path); source.close()

with tempfile.TemporaryDirectory() as temp, patch('heightmap_studio.ConfigStore',return_value=ConfigStore(temp)):
    root=tk.Tk(); ui=app.Application(root)
    ui.image,ui.gray,ui.dpi=app.read_source(path)
    ui.path=path; ui.original_size=native_size; ui.active_mode=2
    ui.raw=np.full_like(ui.gray,.6)
    root.update(); ui.render(); root.update()
    ui.working(False)
    assert ui.native_source is None
    anchor=SimpleNamespace(x=ui.source_canvas.winfo_width()*.6,y=ui.source_canvas.winfo_height()*.55)
    before=ui.image_point(anchor)
    ui.zoom_source(2,anchor.x,anchor.y)
    after=ui.image_point(anchor)
    assert np.allclose(before,after,atol=1e-5)
    ui.actual_source(); root.update()
    assert abs(ui.source_transform[2]-1)<1e-6
    started=time.monotonic()
    while ui.native_source is None:
        root.update(); time.sleep(.02)
        assert time.monotonic()-started<15
    assert ui.native_source.getpixel((2051,1500))==(5,20,250)
    ui.pan_start(SimpleNamespace(x=200,y=200))
    old=ui.source_center
    ui.pan_move(SimpleNamespace(x=230,y=180))
    assert np.allclose(ui.source_center,(old[0]-30,old[1]+20))
    ui.pan_end(SimpleNamespace())
    ui.source_center=(2050,1536)
    ui.zoom_source(absolute=4)
    ui.redraw_source(); root.update()
    ox,oy,scale=ui.source_transform
    point=(2050.0,1536.0)
    event=SimpleNamespace(x=ox+point[0]*scale,y=oy+point[1]*scale)
    ui.brush_mode.set('Erase background'); ui.brush_size.set(8)
    ui.paint_start(event); ui.paint_end(event)
    stroke=ui.strokes[0]
    assert np.allclose(stroke['points'][0],point)
    assert abs(stroke['radius']-1)<1e-6
    base,mask=app.compose(ui.raw,ui.image,app.Settings(smooth=0),2,ui.gray)
    out=folder/'native_precise_mask.png'
    app.export_native(out,base,mask,path,native_size,16,manual_strokes=ui.strokes)
    with Image.open(out) as result:
        result.load()
        assert result.getpixel((2050,1536))==0
        assert result.getpixel((2053,1536))>0
    gui_out=folder/'GUI_native_precise_mask.png'
    with patch('tkinter.filedialog.asksaveasfilename',return_value=str(gui_out)),patch('tkinter.messagebox.showinfo'):
        ui.export()
        started=time.monotonic()
        while ui.busy:
            root.update(); time.sleep(.02)
            assert time.monotonic()-started<30
    with Image.open(gui_out) as result:
        assert result.size==native_size
        assert result.getpixel((2050,1536))==0
        assert result.getpixel((2053,1536))>0
    ui.undo_stroke(); assert not ui.strokes
    ui.paint_start(event); ui.paint_end(event)
    ui.brush_mode.set('Restore mask'); ui.paint_start(event); ui.paint_end(event)
    restored=app.raster_strokes(Image.new('L',native_size,255),ui.strokes)
    assert restored.getpixel((2050,1536))==255
    ui.clear_mask()
    ui.fit_source(); root.update()
    assert abs(ui.source_zoom-1)<1e-6
    ui.native_source.close()
    root.destroy()

report={'version':app.VERSION,'cursor_anchored_zoom':'passed','one_to_one_native_view':'passed',
        'native_detail_loaded':'passed','pan_transform':'passed','brush_size_constant_on_screen':'passed',
        'native_mask_center_and_nearby_pixel':'passed','undo_restore_after_zoom':'passed','fit_reset':'passed'}
(folder/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
