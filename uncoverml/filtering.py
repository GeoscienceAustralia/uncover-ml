"""Code for computing the gamma sensor footprint, and for applying and
unapplying spatial convolution filters to a given image.

BM: this is used in scripts/gammasensor_cli.py - I haven't used it in 
my time with uncoverml or seen it used. 
"""

import numpy as np
import numpy.fft as fft
from skimage.restoration import deconvolution


def pad2(img):
    img = np.ma.vstack((img, img[-2::-1]))
    img = np.ma.hstack((img, img[:, -2::-1]))
    return img


def fwd_filter(img, S):
    img_w, img_h, ch = img.shape
    F = pad2(img)
    F.data[F.mask] = 0.  # make sure its zero-filled!

    # Forward transform
    specF = np.fft.fft2(F.data.astype(float), axes=(0, 1))
    specN = np.fft.fft2(1. - F.mask.astype(float), axes=(0, 1))
    specS = np.fft.fft2(S[::-1, ::-1])
    out = np.real(np.fft.ifft2(specF * specS[:, :, np.newaxis], axes=(0, 1)))
    norm = np.real(np.fft.ifft2(specN * specS[:, :, np.newaxis], axes=(0, 1)))
    eps = 1e-15
    norm = np.maximum(norm, eps)
    out /= norm
    out = out[-img_w:, -img_h:]
    out[img.mask] = 0.
    return np.ma.MaskedArray(data=out, mask=img.mask)


def kernel_impute(img, S):
    F = pad2(img)
    F.data[F.mask] = 0.  # make sure its zero-filled!
    img_w, img_h, img_ch = img.shape
    Q = S
    specF = np.fft.fft2(F.data.astype(float), axes=(0, 1))
    specN = np.fft.fft2(1. - F.mask.astype(float), axes=(0, 1))
    specQ = np.fft.fft2(Q[::-1, ::-1])
    numer = np.real(np.fft.ifft2(specF * specQ[:, :, np.newaxis], axes=(0, 1)))
    denom = np.real(np.fft.ifft2(specN * specQ[:, :, np.newaxis], axes=(0, 1)))
    eps = 1e-15
    fill = numer/(denom+eps)
    fill = fill[-img_w:, -img_h:]

    image = img.data.copy()

    # img = img.copy()
    image[img.mask] = fill[img.mask]
    mask = np.zeros_like(img.mask, dtype=bool)
    return np.ma.MaskedArray(data=image, mask=mask)


def inv_filter(img, S, noise=0.001):

    # If you want to call inv_filter, you have to impute first to avoid
    # spectral artifacts
    assert(np.all(img.mask == False))

    # unfortunately scipy deconvolution messes with our scale...
    F = pad2(img)

    # Do my scaling here to keep scipy happy!
    machine_scale = F.max()
    F /= machine_scale

    S = S / S.sum()  # avoids any scaling effects

    img_w, img_h, img_ch = img.shape
    out = np.zeros_like(F)
    for ch in range(img_ch):
        out[:, :, ch] = deconvolution.wiener(F[:, :, ch].data.astype(float),
                                             S, noise)

    out = machine_scale * out[:img_w, :img_h]
    # out[img.mask] = 0.
    return np.ma.masked_array(out, mask=img.mask)


def sensor_footprint(img_w, img_h, res_x, res_y, height, mu_air):

    x = np.arange(-img_w+1, img_w) * res_x
    y = np.arange(-img_h+1, img_h) * res_y

    yy, xx = np.meshgrid(y, x)
    r = np.sqrt(xx**2 + yy**2 + height**2)
    sens = np.exp(-mu_air*r) / r**2

    # Unknown gain problem
    sens /= sens.max()

    return sens
