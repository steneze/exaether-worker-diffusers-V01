"""
Image utilities for diffusers worker.

resize_to_mpixels, prepare_mask, base64 encode/decode.
"""
import base64
import io
import random

from PIL import Image, ImageFilter


def decode_base64_image(b64: str) -> Image.Image:
    """Decode a base64 string (with or without data URI prefix) to PIL Image."""
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def encode_image(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resize_to_mpixels(img: Image.Image, mpixels: float, align: int = 64) -> Image.Image:
    """Resize preserving aspect ratio, align dimensions to `align` px."""
    w, h = img.size
    current_mp = (w * h) / 1_000_000
    if current_mp <= mpixels * 1.05:
        new_w = (w // align) * align
        new_h = (h // align) * align
        if (new_w, new_h) != (w, h):
            return img.resize((new_w, new_h), Image.LANCZOS)
        return img
    scale = (mpixels / current_mp) ** 0.5
    new_w = (int(w * scale) // align) * align
    new_h = (int(h * scale) // align) * align
    return img.resize((new_w, new_h), Image.LANCZOS)


def prepare_mask(
    mask: Image.Image, target_size: tuple, grow: int = 10, blur: int = 10
) -> Image.Image:
    """Resize, grow, and blur mask. White = zone to regenerate."""
    mask = mask.convert("L")
    mask = mask.resize(target_size, Image.LANCZOS)
    if grow > 0:
        mask = mask.filter(ImageFilter.MaxFilter(size=grow * 2 + 1))
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))
    return mask


def resolve_seed(seed: int) -> int:
    """Resolve seed: -1 means random."""
    if seed is None or seed < 0:
        return random.randint(0, 2**32 - 1)
    return seed
