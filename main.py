import cv2, utils, hdr, matplotlib.pyplot as plt
import numpy as np
import sys
def exposure_time_txts(exposure_times):
    return ['t=%.3f' % t for t in exposure_times]

def tone_map_params_enumeration(hdr_image):
    ldr_images = []
    param_txts = []
    for m in np.linspace(0.45, 0.5, 1):
        for b in np.linspace(0.05, 0.05, 1):
            for g in np.linspace(0.05, 0.05, 1):
                for r in np.linspace(0.05, 0.15, 1):
                    ldr_images.append(hdr.tone_map_reinhard(hdr_image, np.array([m + b, m + g, m + r])))
                    param_txts.append('b=%.2f g=%.2f r=%.2f' % (m+b,m+g,m+r))

    return ldr_images, param_txts


if __name__ == '__main__':
    ROOT_DIR = sys.argv[1]
    images, exposure_times = utils.exposure_series(ROOT_DIR, scale=0.4)
    # exposure_times = np.loadtxt(ROOT_DIR + 'exposure_times.txt', delimiter=',')

    # merge_mertens = cv2.createMergeMertens()
    # res_mertens = merge_mertens.process(images)
    # res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    # utils.show_image(res_mertens_8bit)

    hdr_image = hdr.hdr(images, exposure_times, algorithm=sys.argv[2])
    # cv2.imwrite('result.hdr', hdr_image)
    # hdr_image = cv2.imread(ROOT_DIR + 'result.hdr')

    # np.save(ROOT_DIR + 'result', hdr_image)
    # hdr_image = np.load(ROOT_DIR+'result.npy')

    # utils.show_hdr_image(hdr_image)

    ldr_images, param_txts = tone_map_params_enumeration(hdr_image)
    # ldr_images.append(res_mertens_8bit)
    utils.show_images(ldr_images, cols=1)

    # cv2.imwrite(ROOT_DIR + 'ldr_result.jpg', ldr_image)









