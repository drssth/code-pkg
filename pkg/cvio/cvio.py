import os
import os.path
import cv2
import numpy as np


def resize_image_batch(image_batch, rsh, rsw):
    nc, nh, nw = image_batch.shape[-3:]
    nx = image_batch.shape[:-3]
    nxx = nc
    for i in nx:
        nxx = nxx * i
    item_reshape = image_batch.reshape(nxx, nh, nw)
    interpolate = cv2.INTER_CUBIC
    if max(rsh, rsw) > max(nh, nw):
        interpolate = cv2.INTER_LINEAR
    else:
        interpolate = cv2.INTER_AREA
    item_resize = np.array([cv2.resize(item_reshape[ii], (rsw, rsh), interpolation=interpolate) for ii in range(nxx)])
    item_resize = item_resize.reshape(*nx, nc, rsh, rsw)
    return item_resize



def resize_image_width_height(img, rh, rw):
    nh, nw = img.shape
    interpolate = cv2.INTER_CUBIC
    if max(rh, rw) > max(nh, nw):
        interpolate = cv2.INTER_LINEAR
    else:
        interpolate = cv2.INTER_AREA
    return cv2.resize(img, (rw, rh), interpolation=interpolate)

