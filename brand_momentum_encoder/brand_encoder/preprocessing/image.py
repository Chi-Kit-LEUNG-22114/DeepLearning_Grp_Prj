from pathlib import Path
from PIL import Image, ImageOps
def load_rgb(path:str|Path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")
