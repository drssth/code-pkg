import cv2
import numpy as np


def resize_data(item, rsh, rsw):
    nc, nh, nw = item.shape[-3:]
    nx = item.shape[:-3]
    nxx = nc
    for i in nx:
        nxx = nxx * i
    item_reshape = item.reshape(nxx, nh, nw)
    interpolate = cv2.INTER_CUBIC
    if max(rsh, rsw) > max(nh, nw):
        interpolate = cv2.INTER_LINEAR
    else:
        interpolate = cv2.INTER_AREA
    item_resize = np.array([cv2.resize(
        item_reshape[ii], (rsw, rsh), interpolation=interpolate) for ii in range(nxx)])
    item_resize = item_resize.reshape(*nx, nc, rsh, rsw)
    return item_resize
