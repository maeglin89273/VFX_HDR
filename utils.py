import cv2, matplotlib.pyplot as plt, numpy as np
import os
from math import ceil
def exposure_series(dir):
    cat_dir = lambda filename: dir + filename

    image_filenames = [cat_dir(filename) for filename in os.listdir(dir) if filename[0].isdigit()]
    image_filenames.sort()
    images = [cv2.imread(filename) for filename in image_filenames]
    return images

def bgr_to_rgb(image):
    b, g, r = cv2.split(image)
    return cv2.merge([r, g, b])

def show_hdr_image(hdr_image):
    channel = hdr_image.shape[2]
    show_images([np.log(hdr_image[:, :, i] + np.finfo(np.float32).eps) for i in range(channel)], ['Blue', 'Green', 'Red'], cols=channel, cmap='jet')

def show_image(cv_image, cmap='gray'):
    if cv_image.ndim > 2:
        plt.imshow(bgr_to_rgb(cv_image))
    else:
        plt.imshow(cv_image, cmap=cmap)
    plt.xticks([]), plt.yticks([])
    plt.show()

def show_images(cv_images, param_txts=None, cols=2, cmap='gray'):
    rows = int(ceil(len(cv_images) / cols))
    for i, cv_image in enumerate(cv_images):
        plt.subplot(rows, cols, i + 1)
        if cv_image.ndim > 2:
            plt.imshow(bgr_to_rgb(cv_image))
        else:
            plt.imshow(cv_image, cmap=cmap)

        if param_txts:
            plt.title(param_txts[i], fontsize=8)
        plt.xticks([]), plt.yticks([])

    if param_txts:
        plt.subplots_adjust(left=0.01, right=0.99, bottom= 0.01, top=0.95, wspace=0.02, hspace=0.2)
    else:
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.02, hspace=0.02)

    plt.show()

def show_crf_curves(g_funcs):
    x = np.arange(0, 256)

    for g_func, c in zip(g_funcs, ['b', 'g', 'r']):
        # plt.plot(np.log(poly(x)), x)
        plt.plot(x, g_func, c)

    plt.show()