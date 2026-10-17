# Changelog

## 1.5.0

- Added general outline-based illustration relief, using connected regions and independent distance-based inflation without manually positioned shapes.
- Added an adjustable dark-outline threshold and shallow line grooves.
- Removed near-transparent dust and tiny detached mask islands in illustration mode; silhouette bevels use alpha >127.
- Kept illustration segmentation stable when resizing previews.
- Added synthetic regression fixtures for translation, rotation, different dimensions, line thresholds and masks.

## 1.4.0

- Added source zoom at the cursor, Fit / 100% controls and drag-to-pan.
- Load native source detail in the background when zooming into large images.
- Preserve manual erase and restore strokes in original source coordinates for precise full-size PNG export.
- Keep the status bar visible on shorter windows at Windows display scaling of 125%.

## 1.3.0

- Added optional Hugging Face, OpenAI and Gemini services.
- Added separate account settings with optional Windows DPAPI credential storage.
- Added explicit online generation and connection checks without automatic image uploads.

## 1.2.0

- Added large-image support and native-resolution PNG export in strips.
- Improved preservation of existing grayscale maps and 16-bit precision.

## Initial version

- Offline AI depth estimation, image/map previews, height controls and PNG export.
- Automatic background removal and manual erase/restore tools.
