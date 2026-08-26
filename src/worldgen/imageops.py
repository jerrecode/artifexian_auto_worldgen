from __future__ import annotations

from pathlib import Path


def _pillow_resample(name: str):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "PNG upscaling requires Pillow. Install with: pip install 'artifexian-auto-worldgen[render]'"
        ) from exc
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    try:
        return Image, mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown resampling mode {name!r}; choose from {sorted(mapping)}") from exc


def upscale_png(
    path: str | Path,
    *,
    scale: float,
    resample: str = "lanczos",
    max_megapixels: float = 120.0,
) -> Path:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    path = Path(path)
    if scale == 1.0:
        return path
    Image, mode = _pillow_resample(resample)
    with Image.open(path) as image:
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        mp = (width * height) / 1_000_000.0
        if mp > max_megapixels:
            ratio = (max_megapixels / mp) ** 0.5
            width = max(1, int(width * ratio))
            height = max(1, int(height * ratio))
        resized = image.resize((width, height), resample=mode)
        tmp = path.with_name(path.name + ".upscale.tmp")
        resized.save(tmp, format="PNG", optimize=False)
        tmp.replace(path)
    return path


def list_pngs(root: str | Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(root).rglob("*.png")))


def upscale_png_tree(
    root: str | Path,
    *,
    scale: float,
    resample: str = "lanczos",
    max_megapixels: float = 120.0,
    executor=None,
) -> tuple[Path, ...]:
    paths = list_pngs(root)
    if scale == 1.0 or not paths:
        return paths

    def run(path: Path) -> Path:
        return upscale_png(path, scale=scale, resample=resample, max_megapixels=max_megapixels)

    if executor is None:
        return tuple(run(path) for path in paths)
    return tuple(executor.map(run, paths))
