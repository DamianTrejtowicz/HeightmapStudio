"""General fixtures only: no example-specific coordinates in the algorithm."""
import numpy as np
from PIL import Image, ImageDraw
from outline_relief import outline_relief, clean_support
from heightmap_studio import Settings, compose, source_mask


def fixture(size=(320, 240), offset=(0, 0), outline=(0, 0, 0, 255)):
    image = Image.new('RGBA', size)
    draw = ImageDraw.Draw(image)
    dx, dy = offset
    draw.rounded_rectangle((30+dx, 25+dy, 150+dx, 180+dy), 20,
                           fill=(220, 160, 70, 255), outline=outline, width=5)
    draw.ellipse((180+dx, 80+dy, 220+dx, 120+dy), fill='white', outline=outline, width=4)
    return image


def tests():
    source = fixture()
    height, mask, report = outline_relief(source)
    assert report['regions'] == 2
    assert height[95, 90] > height[95, 31] + .25
    assert height[100, 200] > height[100, 181] + .25  # Small region rises independently.
    assert np.all(height[mask == 0] == 0)
    # Translating the drawing translates the relief, with no anatomy-specific tuning.
    shifted, _, _ = outline_relief(fixture(offset=(20, 15)))
    assert np.allclose(height[20:190, 20:230], shifted[35:205, 40:250], atol=1e-6)
    # Rotating the image rotates the segmentation and gradient field.
    rotated, _, _ = outline_relief(source.transpose(Image.Transpose.ROTATE_180))
    assert np.allclose(height, rotated[::-1, ::-1], atol=1e-6)
    # Other sizes/aspect ratios remain finite and registered, including no regions.
    for size in ((120, 350), (640, 480), (53, 49)):
        image = Image.new('RGBA', size)
        ImageDraw.Draw(image).ellipse((4, 4, size[0]-5, size[1]-5), fill='white', outline='black', width=3)
        z, m, _ = outline_relief(image)
        assert z.shape == (size[1], size[0]) and np.isfinite(z).all()
        assert z.max() > .2 and np.all(z[m == 0] == 0)
    z, _, info = outline_relief(Image.new('RGBA', (10, 10)))
    assert info['regions'] == 0 and not z.any()
    # Dust removal keeps real detached objects, including small parts of a drawing.
    noisy = np.asarray(source).copy()
    noisy[5, 5] = (255, 255, 255, 255)
    noisy[210, 250] = (255, 255, 255, 4)
    cleaned = clean_support(noisy[:, :, 3] / 255)
    assert cleaned[5, 5] == cleaned[210, 250] == 0 and cleaned[100, 200] == 1
    # Medium-gray line art can use a different threshold, without editing source code.
    gray = fixture(outline=(110, 110, 110, 255))
    low, _, _ = outline_relief(gray, threshold=70)
    high, _, _ = outline_relief(gray, threshold=140)
    assert not np.allclose(low, high)
    # White paper removal and the standard manual mask work with this mode too.
    paper = Image.new('RGBA', source.size, 'white'); paper.alpha_composite(source)
    settings = Settings(white_bg=True, smooth=0)
    raw, _, _ = outline_relief(paper, source_mask(paper, settings))
    manual = Image.new('L', paper.size, 255)
    ImageDraw.Draw(manual).rectangle((80, 80, 100, 100), fill=0)
    result, result_mask = compose(raw, paper, settings, 3, manual_mask=manual)
    assert result[0, 0] == result[90, 90] == 0
    assert result_mask[100, 200] == 1 and result[100, 200] > .2
    print('PASS: independent regions, translation/rotation, varied sizes, dust, threshold, paper and manual masks')


if __name__ == '__main__':
    tests()
