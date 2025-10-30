
# --- sitecustomize: PNG->JPG fallback for imread family ---
import os as _os

def _coerce_path(p):
    # Handle pathlib.Path, numpy.str_, bytes; always return str
    try:
        from numpy import str_ as _npstr
    except Exception:
        _npstr = ()
    if isinstance(p, (_os.PathLike,)):
        p = _os.fspath(p)
    if isinstance(p, bytes):
        p = p.decode("utf-8", "ignore")
    if _npstr and isinstance(p, _npstr):
        p = str(p)
    return p

def _with_png2jpg_fallback(reader):
    def _wrapped(path, *a, **k):
        path = _coerce_path(path)
        try:
            return reader(path, *a, **k)
        except FileNotFoundError:
            if isinstance(path, str) and path.lower().endswith(".png"):
                alt = path[:-4] + ".jpg"
                if _os.path.exists(alt):
                    return reader(alt, *a, **k)
            raise
    return _wrapped

# Patch imageio v3, v2, PIL
try:
    import imageio.v3 as _iio3
    _iio3.imread = _with_png2jpg_fallback(_iio3.imread)
    print("[png2jpg] patched imageio.v3.imread")
except Exception as e:
    print("[png2jpg] imageio.v3 patch skipped:", repr(e))

try:
    import imageio as _iio
    _iio.imread = _with_png2jpg_fallback(_iio.imread)
    print("[png2jpg] patched imageio.v2.imread")
except Exception as e:
    print("[png2jpg] imageio.v2 patch skipped:", repr(e))

try:
    from PIL import Image as _PILImage
    _orig_open = _PILImage.open
    def _open_wrap(p, *a, **k):
        p = _coerce_path(p)
        try:
            return _orig_open(p, *a, **k)
        except FileNotFoundError:
            if isinstance(p, str) and p.lower().endswith(".png"):
                alt = p[:-4] + ".jpg"
                if _os.path.exists(alt):
                    return _orig_open(alt, *a, **k)
            raise
    _PILImage.open = _open_wrap
    print("[png2jpg] patched PIL.Image.open")
except Exception as e:
    print("[png2jpg] PIL patch skipped:", repr(e))

# Patch skimage.io.imread AND the imageio plugin alias inside skimage
try:
    from skimage import io as _sio
    _sio.imread = _with_png2jpg_fallback(_sio.imread)
    print("[png2jpg] patched skimage.io.imread")
    try:
        import skimage.io._plugins.imageio_plugin as _io_plugin
        if hasattr(_io_plugin, "imageio_imread"):
            _io_plugin.imageio_imread = _with_png2jpg_fallback(_io_plugin.imageio_imread)
            print("[png2jpg] patched skimage imageio_plugin.imageio_imread")
    except Exception as e:
        print("[png2jpg] skimage plugin patch skipped:", repr(e))
except Exception as e:
    print("[png2jpg] skimage patch skipped:", repr(e))

print("[sitecustomize] PNG->JPG fallback active")


# --- allowlist xrv DenseNet for torch.load with weights_only=True ---
try:
    import torch, torch.serialization
    import torchxrayvision.models as _xrv_models
    torch.serialization.add_safe_globals([_xrv_models.DenseNet])
    print("[sitecustomize] allowlisted torchxrayvision.models.DenseNet for safe load")
except Exception as e:
    print("[sitecustomize] allowlist skipped:", repr(e))

# --- torchvision ToPILImage guard: coerce to <=4ch (prefer 1ch) ---
try:
    import numpy as _np
    import torch as _torch
    import torchvision.transforms.functional as _F
    from torchvision.transforms import ToPILImage as _ToPILImage

    _real_to_pil = _F.to_pil_image
    def _coerce_1ch(x):
        # Accept torch.Tensor or np.ndarray. Return same type, forced to single channel.
        if isinstance(x, _torch.Tensor):
            arr = x
            if arr.ndim >= 3:
                # move channel to last if it's first: (C,H,W)->(H,W,C)
                if arr.ndim == 3 and arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
                    arr = arr.permute(1,2,0)
                # squash to (H,W,C) then take first channel
                arr = arr.reshape(arr.shape[0], arr.shape[1], -1) if arr.ndim==3 else arr
                if arr.ndim > 3:
                    arr = arr.view(arr.shape[0], arr.shape[1], -1)
                if arr.shape[-1] > 1:
                    arr = arr[..., 0]
            return arr
        elif isinstance(x, _np.ndarray):
            arr = x
            if arr.ndim >= 3:
                if arr.ndim > 3:
                    arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
                if arr.shape[-1] > 1:
                    arr = arr[..., 0]
            return arr
        else:
            return x

    def _to_pil_guard(pic, mode=None):
        try:
            return _real_to_pil(pic, mode)
        except Exception:
            pic2 = _coerce_1ch(pic)
            return _real_to_pil(pic2, mode)

    _F.to_pil_image = _to_pil_guard

    # Also guard the transform class entry point
    _real_call = _ToPILImage.__call__
    def __call__(self, pic):
        try:
            return _real_call(self, pic)
        except Exception:
            return _real_call(self, _coerce_1ch(pic))
    _ToPILImage.__call__ = __call__

    print("[sitecustomize] ToPILImage guard active (force <=1ch on error)")
except Exception as _e:
    print("[sitecustomize] ToPILImage guard failed to load:", _e)
# --- end guard ---
