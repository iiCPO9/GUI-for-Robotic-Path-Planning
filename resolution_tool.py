import os
from PIL import Image

TARGET_W, TARGET_H = 298, 169

def decrease_resolution(path, target_w=TARGET_W, target_h=TARGET_H):
    """Decrease resolution of an image to target_w x target_h if needed."""
    img = Image.open(path).convert("RGB")
    width, height = img.size

    # Already exact
    if width == target_w and height == target_h:
        return path

    # Use min() to ensure we fit within target without upscaling
    scale = min(target_w / width, target_h / height)
    
    # If scale >= 1, image is already smaller than target, just return
    if scale >= 1.0:
        return path

    new_w, new_h = int(width * scale), int(height * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to exact target size
    left = (resized.size[0] - target_w) // 2
    top = (resized.size[1] - target_h) // 2
    cropped = resized.crop((left, top, left + target_w, top + target_h))

    # Save new file next to original
    base, ext = os.path.splitext(path)
    out_path = base + f"_resized{ext}"
    cropped.save(out_path)

    return out_path