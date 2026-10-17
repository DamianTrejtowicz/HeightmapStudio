import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageGrab
import heightmap_studio as app


def tests():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        values = np.arange(256, dtype=np.uint8).reshape(16, 16)
        Image.fromarray(values).save(d/'źródło.png')
        im, gray, dpi = app.read_source(d/'źródło.png')
        height, mask = app.compose(gray, im, app.Settings(smooth=0), 2, gray)
        app.export_height(d/'mapa.png', height, mask)
        assert np.array_equal(np.asarray(Image.open(d/'mapa.png')), values)
        original16 = np.arange(256, dtype=np.uint16).reshape(16,16)
        Image.fromarray(original16).save(d/'16.png')
        im16, raw16, _ = app.read_source(d/'16.png')
        app.export_height(d/'roundtrip16.png', raw16, np.ones_like(raw16), 16)
        assert np.array_equal(np.asarray(Image.open(d/'roundtrip16.png')), original16)
        inv, _ = app.compose(gray, im, app.Settings(smooth=0, invert=True), 2, gray)
        assert np.allclose(inv + height, 1)
        rgba = np.full((32,32,4), 255, np.uint8)
        rgba[:8,:,3] = 0
        transparent = Image.fromarray(rgba)
        z, m = app.compose(np.ones((32,32),np.float32)*.5, transparent, app.Settings(), 2)
        assert np.all(z[:8] == 0)
        app.export_height(d/'alpha.png',z,m,8,True)
        assert Image.open(d/'alpha.png').mode == 'LA'
        assert np.array_equal(np.asarray(Image.open(d/'alpha.png'))[:,:,1],rgba[:,:,3])
        a = np.full((40,40,4),255,np.uint8)
        a[10:30,10:30,:3]=[230,40,130]
        # Neutral shadow touching the outer white field must also disappear.
        a[28:34,8:32,:3]=210
        m = app.source_mask(Image.fromarray(a), app.Settings(white_bg=True))
        assert m[0,0] == 0 and m[20,20] == 1 and m[32,20] == 0
        manual=Image.new('L',im.size,255)
        from PIL import ImageDraw
        ImageDraw.Draw(manual).rectangle((4,4,10,10),fill=0)
        manual_height,manual_alpha=app.compose(gray,im,app.Settings(smooth=0),2,gray,manual)
        assert np.all(manual_height[4:11,4:11]==0)
        assert np.all(manual_alpha[4:11,4:11]==0)
        app.export_height(d/'manual.png',manual_height,manual_alpha,16)
        assert np.all(np.asarray(Image.open(d/'manual.png'))[4:11,4:11]==0)
    print('PASS: grayscale roundtrip, low-valued 16-bit precision, inversion, alpha, white background + shadows')


if __name__ == '__main__':
    tests()
