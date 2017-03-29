import cv2, utils, hdr, matplotlib.pyplot as plt
import imageio
import numpy as np
import sys
import os


if __name__ == '__main__':
    ROOT_DIR = sys.argv[1]
    images, exposure_times = utils.exposure_series(ROOT_DIR, scale=1)

    #load from .hdr file
    # hdr_image = cv2.imread(os.path.join(ROOT_DIR, 'hdr_result.hdr'), -1)
    hdr_image = hdr.hdr(images, exposure_times, algorithm=sys.argv[2])
    utils.show_hdr_image(hdr_image)
    #save .hdr file
    # cv2.imwrite(os.path.join(ROOT_DIR, 'hdr_result.hdr'), hdr_image)

    a = 0.55
    ldr_image = hdr.tone_map_reinhard(hdr_image, np.array([a] * 3))
    utils.show_images([ldr_image], cols=1)

    #save the ldr image
    # cv2.imwrite(os.path.join(ROOT_DIR,'ldr_result.jpg'), np.clip(ldr_images_2[0] * 255, 0, 255).astype('uint8'))








