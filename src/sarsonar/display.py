'''
Display compression: raw amplitude images -> dB, asinh or linear arrays, 8-bit grayscale and PNG.

The renderers return raw amplitude, which spans ~100 dB in a sonar image, so a linear stretch
shows little but specular glints; dB and asinh are what show the seafloor and shadows.
'''
import numpy as np
import torch

COMPRESSIONS = ('db', 'asinh', 'linear')


def db_compress(amplitude, reference, db_floor=-60.0):
    '''
    Amplitude -> dB below reference, clipped to [db_floor, 0].

    inputs:
        amplitude (ndarray): raw amplitude, any shape
        reference (float): amplitude that maps to 0 dB; <= 0 maps everything to db_floor
        db_floor (float): black point
    '''
    amplitude = np.asarray(amplitude, dtype=np.float32)
    if reference <= 0.0:
        return np.full_like(amplitude, db_floor)
    floor_linear = 10.0 ** (db_floor / 20.0)
    db = 20.0 * np.log10(np.clip(amplitude / reference, floor_linear, None))
    return np.clip(db, db_floor, 0.0)


def asinh_compress(amplitude, k, ref):
    '''
    arcsinh(amplitude/k) / arcsinh(ref/k), clipped to [0,1]: linear below k, ~logarithmic above,
    so faint texture survives without dB's hard floor.
    '''
    amplitude = np.asarray(amplitude, dtype=np.float32)
    return np.clip(np.arcsinh(amplitude / k) / np.arcsinh(ref / k), 0.0, 1.0)


def compress(amplitude, compression='db', db_floor=-60.0, asinh_k_ratio=0.005):
    '''
    One image's display array in [0,1], referenced to that image's own level.

    inputs:
        amplitude (H,W): raw amplitude, ndarray or tensor
        compression (str): 'db' -- dB below the image's peak, floored at db_floor; 'asinh' --
            asinh_compress referenced to the image's 99.9th percentile, with k = asinh_k_ratio * ref;
            'linear' -- amplitude / peak
    outputs:
        display (H,W) float32 in [0,1]
    '''
    if isinstance(amplitude, torch.Tensor):
        amplitude = amplitude.detach().abs().cpu().numpy()
    amplitude = np.abs(np.asarray(amplitude, dtype=np.float32))
    peak = float(amplitude.max()) if amplitude.size else 0.0

    if compression == 'db':
        return (db_compress(amplitude, peak, db_floor) - db_floor) / -db_floor
    if compression == 'asinh':
        ref = float(np.percentile(amplitude, 99.9))
        if ref <= 0.0:
            return np.zeros_like(amplitude)
        return asinh_compress(amplitude, asinh_k_ratio * ref, ref)
    if compression == 'linear':
        return amplitude / peak if peak > 0.0 else np.zeros_like(amplitude)
    raise ValueError('compression must be one of %s, got %r' % (COMPRESSIONS, compression))


def to_uint8(amplitude, compression='db', db_floor=-60.0, asinh_k_ratio=0.005):
    '''compress, scaled to a uint8 [0,255] grayscale image.'''
    return (compress(amplitude, compression, db_floor, asinh_k_ratio) * 255.0).round().astype(np.uint8)


def save_png(amplitude, path, compression='db', db_floor=-60.0, asinh_k_ratio=0.005):
    '''Write a raw amplitude image to path as an 8-bit grayscale PNG.'''
    from PIL import Image
    Image.fromarray(to_uint8(amplitude, compression, db_floor, asinh_k_ratio)).save(path)
