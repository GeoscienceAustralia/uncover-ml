# Applies S or S^{-1} to a given image
import matplotlib.pyplot as pl
import numpy as np
import numpy.fft as fft
from scipy.ndimage import imread  # needs pillow!
import os

test_file = os.path.expanduser('~/Data/badger.jpg')

def main():

    gain = 1.
    mass_attenuation_air = 0.09  # assume some sort of bulk property
    density_air = 1.22
    mu_air = mass_attenuation_air * density_air

    res_x = 20.0  # metres
    res_y = 20.0

    # Note: images will come in wid*hei format
    img0 = np.mean(imread(test_file), axis=2).T

    img = img0
    img_w, img_h = img.shape

    height = 100  # metres
    S = sensor_footprint(img_w, img_h, res_x, res_y, height, mu_air, gain)
   
    out1 = fwd_filter(img, S)
    out2 = inv_filter(out1, S)

    pl.subplot(131)
    pl.imshow(img.T, interpolation='none', cmap=pl.cm.gray)
    pl.title('Original')
    pl.subplot(132)
    pl.imshow(out1.T, interpolation='none', cmap=pl.cm.gray)
    pl.title('Forward Sensored')
    pl.subplot(133)
    pl.imshow(out2.T, interpolation='none', cmap=pl.cm.gray)
    pl.title('Inverse Sensored')
    pl.show()


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


if __name__=='__main__':
    main()
