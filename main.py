import cv2, utils, hdr, matplotlib.pyplot as plt
import numpy as np
ROOT_DIR = 'samples/exposures/'
EXPOSURE_TIME = [13, 10, 4, 3, 2, 1, 0.3333333333333333, 0.25, 0.016666666666666666, 0.0125, 0.003125, 0.0025, 0.001]
# EXPOSURE_TIME = [15.0, 2.5, 0.25, 0.0333]
def exposure_time_txts(exposure_times):
    return ['t=%.3f' % t for t in exposure_times]

def tone_map_basic_params_enumeration(hdr_image):
    ldr_images = []
    param_txts = []
    for m in np.linspace(0.2, 0.6, 12):
        for b in np.linspace(0.05, 0.05, 1):
            for g in np.linspace(0.05, 0.05, 1):
                for r in np.linspace(0.05, 0.15, 1):
                    ldr_images.append(hdr.tone_map_basic(hdr_image, np.array([m + b, m + g, m + r])))
                    # param_txts.append('g=%.2f l=%.2f c=%.2f' % (m+b,m+g,m+r))

    return ldr_images, param_txts


if __name__ == '__main__':
    images = utils.exposure_series(ROOT_DIR)
    exposure_times = np.array(EXPOSURE_TIME, dtype=np.float32)

    # merge_mertens = cv2.createMergeMertens()
    # res_mertens = merge_mertens.process(images)
    # res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    # utils.show_image(res_mertens_8bit)

    # hdr_image = hdr.hdr(images, exposure_times)
    # print(np.mean(hdr_image))
    # utils.show_hdr_image(hdr_image)
    # np.save(ROOT_DIR + 'result', hdr_image)

    # print("%f, %f" % (np.max(hdr_image), np.min(hdr_image)))
    # ldr_images, param_txts = tone_mapping_params_enumeration(hdr_image)
    # utils.show_images(ldr_images, param_txts, 1)


    # cv2.imwrite(ROOT_DIR + 'ldr_result.jpg', ldr_image)

    hdr_image = np.load(ROOT_DIR+'result.npy')
    # print(np.mean(hdr_image))
    # utils.show_hdr_image(hdr_image)
    ldr_images, param_txts = tone_map_basic_params_enumeration(hdr_image)
    utils.show_images(ldr_images, param_txts, 3)









