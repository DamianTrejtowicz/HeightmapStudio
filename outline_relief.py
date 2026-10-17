"""Image-aligned illustrative relief; no manually positioned geometric volumes."""
import numpy as np
from scipy.ndimage import (binary_closing, distance_transform_edt, find_objects,
                           gaussian_filter, label)


def clean_support(mask):
    """Drop near-transparent dust and tiny isolated islands; keep real detached parts."""
    mask = np.clip(np.asarray(mask, dtype=np.float32), 0, 1)
    components, count = label(mask > 8 / 255)
    if not count:
        return np.zeros_like(mask)
    area = np.bincount(components.ravel())
    minimum = max(4, round(mask.size * 0.000005))
    keep = area >= minimum
    keep[0] = False
    # A deliberately tiny source should still retain its largest visible object.
    if not keep.any():
        keep[1 + np.argmax(area[1:])] = True
    return np.where(keep[components], mask, 0).astype(np.float32)


def outline_relief(image, mask=None, threshold=70):
    """Inflate connected light regions separated by dark strokes.

    The returned height is unmasked; callers apply the returned alpha once.
    Closed regions are disjoint, so no hard maximum of overlapping domes is used.
    This estimates decorative relief, not semantic parts or measured 3D depth.
    """
    rgba = np.asarray(image.convert('RGBA'))
    if mask is None:
        mask = rgba[:, :, 3].astype(np.float32) / 255
    mask = clean_support(mask)
    solid = mask > .5
    if not solid.any():
        return np.zeros_like(mask), mask, {'regions': 0}
    scale = max(image.size) / 2048
    luminance = rgba[:, :, :3].astype(np.float32) @ np.array([.2126, .7152, .0722], np.float32)
    dark = (luminance < threshold) & solid
    # Close one-pixel breaks, retaining image coordinates and the silhouette.
    lines = binary_closing(dark, structure=np.ones((3, 3), bool)) & solid
    lines |= dark
    regions, count = label(solid & ~lines)
    local = np.zeros(mask.shape, np.float32)
    accepted = 0
    for number, box in enumerate(find_objects(regions), 1):
        if box is None:
            continue
        region = regions[box] == number
        area = int(region.sum())
        if area < max(4, round(5 * scale * scale)):
            continue
        distance = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
        radius = float(distance.max())
        if radius <= 0:
            continue
        # Normalize each region independently so fingers and narrow plates rise too.
        t = np.clip(distance / radius, 0, 1)
        dome = np.sin(t * np.pi / 2)
        amplitude = .36 + .14 * (1 - np.exp(-radius / max(12 * scale, 1)))
        tile = local[box]
        tile[region] = (amplitude * dome)[region]
        accepted += 1
    edge = distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1]
    base = .16 + .15 * (1 - np.exp(-edge / max(28 * scale, 1)))
    # Blur the one shared field rather than intersecting hand-placed shapes.
    sigma = max(.6, 2 * scale)
    weight = gaussian_filter(solid.astype(np.float32), sigma)
    inflated = gaussian_filter(local, sigma) / np.maximum(weight, 1e-6)
    groove = gaussian_filter(lines.astype(np.float32), max(.5, .8 * scale))
    height = base + inflated - .055 * groove
    # One inward silhouette bevel from the >127 alpha mask; no second outline ridge.
    t = np.clip(edge / max(10 * scale, 1), 0, 1)
    bevel = t * t * (3 - 2 * t)
    height = np.clip(height * bevel, 0, 1)
    height[~solid] = 0
    return height.astype(np.float32), mask, {
        'regions': accepted, 'dark_threshold': threshold,
        'removed_support_pixels': int(np.count_nonzero((rgba[:, :, 3] > 0) & (mask == 0))),
        'method': 'dark-outline connected regions, per-region distance inflation, shared smoothing and shallow grooves',
    }
