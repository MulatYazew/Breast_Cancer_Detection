# App background image

`app.py` renders a background image behind the clinician review UI, under a
translucent white overlay (`BACKGROUND_OVERLAY_OPACITY` in `app.py`) so
foreground text and controls stay readable. It's picked up automatically from:

```
background.jpg   (or background.jpeg / background.png)
```

If none of those files exist, `background_placeholder.svg` (a simple flat
illustration, not a photo) is used instead.

## Using a real photo

Don't drop a sharp, high-contrast photo in directly — any legible text,
logos/branding, or busy detail in the source photo will show through and
distract from (or look like it belongs to) this app's own UI. `background.jpg`
should be a **pre-softened** version: blurred and desaturated so it reads as
ambient scenery, not competing content. Process a new source photo like this:

```python
from PIL import Image, ImageFilter, ImageEnhance

img = Image.open("your_source_photo.png").convert("RGB")
img = img.filter(ImageFilter.GaussianBlur(radius=22))       # obscure text/logos
img = ImageEnhance.Color(img).enhance(0.75)                  # calmer palette
img = ImageEnhance.Contrast(img).enhance(0.85)
img = ImageEnhance.Brightness(img).enhance(1.12)
img.save("app/assets/background.jpg", "JPEG", quality=82, optimize=True)
```

Re-check the result at actual size before committing to it — blur radius that
looks fine thumbnail-sized can still leave text legible at full resolution.

The original full-resolution photo (if you want to keep one on hand to
reprocess with different settings later) can live alongside as
`background_source.png` — also gitignored, since source photos are typically
licensed stock art that shouldn't be committed.
