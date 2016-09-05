"""Code for computing the gamma sensor footprint, and for applying and
unapplying spatial convolution filters to a given image.
"""

import numpy as np
import numpy.fft as fft


def pad2(img):
    img = np.vstack((img, img[-2::-1]))
    img = np.hstack((img, img[:, -2::-1]))
    return img


def fwd_filter(img, S):
    
    img_w, img_h = img.shape
    F = pad2(img)
    # Forward transform
    specF = np.fft.fft2(F)
    specS = np.fft.fft2(S[::-1, ::-1])
    out = np.real(np.fft.ifft2(specF * specS))
    out = out[-img_w:, -img_h:]
    return out


def inv_filter(img, S):
    img_w, img_h = img.shape
    # Reverse transform - Edge padding unknown might make tiny differences?
    F = pad2(img)
    specF = np.fft.fft2(F)
    specS = np.fft.fft2(S[::-1, ::-1])
    # Is this sensible? maybe gaining energy from outside
    out = np.real(np.fft.ifft2(specF / specS))
    out = out[-img_w:, -img_h:]
    # out = out[:img_w, :img_h]
    return out 



def sensor_footprint(img_w, img_h, res_x, res_y, height, mu_air, gain):
    x = np.arange(-img_w+1, img_w) * res_x
    y = np.arange(-img_h+1, img_h) * res_y
    
    yy, xx = np.meshgrid(y, x)

    # Invoke sensor model
    r = np.sqrt(xx**2 + yy**2 + height**2)
    sens = gain * np.exp(-mu_air*r) / r**2
    return sens

