import cv2, numpy as np, matplotlib.pyplot as plt
import utils

#Implementation of MTB alignment algorithm
def align_images(images, median_margin=2):
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    result = []

    pivot_image_idx = len(gray_images) // 2
    pivot_image = gray_images[pivot_image_idx]
    pyramid_op_num = int(np.floor(np.log2(np.min(pivot_image.shape)))) - 1

    pivot_bitmap_pyramid, eb_pyramid = compute_bitmap_pyramid(pivot_image, pyramid_op_num, median_margin)

    for i, shifted_image in enumerate(gray_images):
        if i == pivot_image_idx:
            result.append(images[i])
            continue

        x, y = compute_full_shift(pivot_bitmap_pyramid, eb_pyramid, shifted_image, pyramid_op_num, median_margin)
        result.append(shift_color_image(images[i], x, y))
        log_alignment_result(x, y)
    return result

def compute_bitmap_pyramid(image, pyramid_op_num, median_margin):
    bitmap, eb = compute_bitmap(image, median_margin)
    bitmap_pyramid = [bitmap]
    eb_pyramid = [eb]
    for i in range(pyramid_op_num):
        image = cv2.pyrDown(image)
        bitmap, eb = compute_bitmap(image, median_margin)
        bitmap_pyramid.append(bitmap)
        eb_pyramid.append(eb)

    return bitmap_pyramid, eb_pyramid

def compute_full_shift(pivot_bitmap_pyramid, pivot_eb_pyramid, shifted_image, pyramid_op_num, median_margin):
    shifted_bitmap_pyramid, shifted_eb_pyramid = compute_bitmap_pyramid(shifted_image, pyramid_op_num, median_margin)
    x = 0
    y = 0
    for (pivot_bitmap, shifted_bitmap, pivot_eb, shifted_eb) in zip(reversed(pivot_bitmap_pyramid), reversed(shifted_bitmap_pyramid),
                                                                    reversed(pivot_eb_pyramid), reversed(shifted_eb_pyramid)):
        x *= 2
        y *= 2
        shifted_bitmap = shift_bitmap(shifted_bitmap, x, y)
        shifted_eb = shift_bitmap(shifted_eb, x, y)
        dx, dy = compute_shift(pivot_bitmap, shifted_bitmap, pivot_eb, shifted_eb)
        x += dx
        y += dy

    return x, y

def compute_shift(pivot_bitmap, shifted_bitmap, pivot_eb, shifted_eb):
    x = 0
    y = 0
    min_error = pivot_bitmap.size
    for ix, iy in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
        ss_bitmap = shift_bitmap(shifted_bitmap, ix, iy)
        ss_eb = shift_bitmap(shifted_eb, ix, iy)
        error = np.count_nonzero(ss_bitmap ^ pivot_bitmap & pivot_eb & ss_eb)
        if min_error > error:
            min_error = error
            x = ix
            y = iy
    return x, y

def shift_bitmap(bitmap, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    h, w = bitmap.shape

    return cv2.warpAffine(bitmap.astype('uint8'), M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype('bool')


def shift_color_image(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    h, w, c = image.shape
    result = np.empty_like(image)

    for ci in range(c):
        result[:, :, ci] = cv2.warpAffine(image[:, :, ci], M, (w, h), borderMode=cv2.BORDER_DEFAULT)
    return result

def log_alignment_result(x, y):
    print('alignment shift: %d, %d' % (x, y))
    if np.abs(x) > 50 or np.abs(y) > 50:
        print('WARNING: This alignment has big shifts. Try to adjust the median margin')


def compute_bitmap(image, median_margin):
    median = np.median(image)
    result = image > median
    exclusive_bitmap = (image > median + median_margin) | (image < median - median_margin)
    return result, exclusive_bitmap


COLOR_N = 256
Z_MID = 127
def hdr_debevec(exposure_images, exposure_times, l=3.5):
    exposure_images = np.array(exposure_images)
    sampling_area = cv2.Canny(cv2.cvtColor(exposure_images[exposure_images.shape[0] // 2], cv2.COLOR_BGR2GRAY), 150, 200) < 127
    channel_images = [exposure_images[:, :, :, c] for c in range(exposure_images.shape[-1])]
    ln_dt = np.log(exposure_times)

    # weight_z = np.array([z if z <= Z_MID else COLOR_N - z for z in range(COLOR_N)])
    weight_z = np.array([z + 1 if z < Z_MID else Z_MID for z in range(COLOR_N)])
    # weight_z = np.array([Z_MID for z in range(COLOR_N)])

    weight_z = weight_z / np.sum(weight_z)

    g_funcs = []
    for c, images in enumerate(channel_images):
        Z = reshape_to_z_ij_and_sample(images, sampling_area)
        A = np.zeros((Z.size + 1 + COLOR_N - 2, COLOR_N + Z.shape[0]))
        b = np.zeros(A.shape[0])
        k = 0
        for i in range(Z.shape[0]):
            for z in range(Z.shape[1]):
                z_ij = Z[i, z]
                w_z_ij = weight_z[z_ij]
                A[k, z_ij] = w_z_ij
                A[k, COLOR_N + i] = -w_z_ij
                b[k] = w_z_ij * ln_dt[z]
                k += 1

        A[k, 127] = 1
        b[k] = 5.5
        for i in range(k + 1, A.shape[0]):
            z = i - (k + 1)
            A[i, z] = l * weight_z[z]
            A[i, z + 1] = -2 * l * weight_z[z + 1]
            A[i, z + 2] = l * weight_z[z + 2]

        vars, error, _, __ = np.linalg.lstsq(A, b)
        g_funcs.append(vars[:COLOR_N])

    utils.show_crf_curves(np.exp(g_funcs))
    hdr_image = debevec_reconstruct_hdr(exposure_images, g_funcs, weight_z, ln_dt)
    return hdr_image

def debevec_reconstruct_hdr(exposure_images, g_funcs, weight_z, ln_dt):
    hdr_image = np.empty(exposure_images.shape[1:])
    channel_images = [exposure_images[:, :, :, c] for c in range(exposure_images.shape[-1])]
    for c, (images, g_func) in enumerate(zip(channel_images, g_funcs)):
        Z = images.reshape((images.shape[0], -1)).T

        ln_E = np.empty(Z.shape[0])
        for i in range(ln_E.size):
            n = 0.0
            d = 0.0
            for j in range(Z.shape[1]):
                n += weight_z[Z[i, j]] * (g_func[Z[i, j]] - ln_dt[j])
                d += weight_z[Z[i, j]]

            ln_E[i] = n / d

        hdr_image[:, :, c] = np.exp(ln_E).reshape(hdr_image.shape[:-1])

    return hdr_image

def reshape_to_z_ij_and_sample(images, sampling_area, samples=100):
    good_sampling_positions = np.where(sampling_area.ravel())[0]
    sampling_positions = good_sampling_positions[np.random.choice(good_sampling_positions.size, samples)]
    images = images.reshape((images.shape[0], -1)).T
    return images[sampling_positions]

def hdr_poly(exposure_images, exposure_times, max_poly_n=5):
    exposure_images = np.array(exposure_images)
    channel_images = [exposure_images[:, :, :, c] for c in range(3)]
    exposure_ratios = compute_exposure_ratios(exposure_times)

    hdr_image = np.empty(exposure_images.shape[1:])
    polys = []
    for c, images in enumerate(channel_images):
        polys.append(find_poly(images, exposure_ratios, max_poly_n))

    utils.show_crf_curves([poly(np.arange(0,255)) for poly in polys])
        #hdr_image[:,:, c] = reconstruct_hdr_intensity(images, coefs)


def find_poly(images, exposure_ratios, max_poly_n):
    min_error = np.inf
    best_coefs = None
    images = reshape_to_z_ij_and_sample(images)
    ratio_vector = np.tile(exposure_ratios, (images.shape[0], 1)).ravel()

    for poly_n in range(2, max_poly_n):
        b = np.zeros(poly_n + 2)
        b[-1] = 1
        a = np.ones((b.shape[0], poly_n + 1))
        for term_i in range(0, poly_n + 1):
            tmp1 = images[:, :-1].ravel() ** term_i - ratio_vector * images[:, 1:].ravel() ** term_i
            tmp1 = tmp1[:, np.newaxis]
            tmp2 = np.empty((tmp1.shape[0], poly_n + 1))
            for term_j in range(poly_n, -1, -1):
                tmp2[:, term_j] = images[:, :-1].ravel() ** term_j - ratio_vector * images[:, 1:].ravel() ** term_j

            a[term_i, :] = np.sum(tmp1 * tmp2, axis=0)
        coefs, error, _, __ = np.linalg.lstsq(a, b)
        if error < min_error:
            min_error = error
            best_coefs = coefs
    return np.poly1d(best_coefs)


def compute_exposure_ratios(exposure_times):
    return np.array([exposure_times[i] / exposure_times[i + 1] for i in range(len(exposure_times) - 1)])

def hdr(exposure_images, exposure_times, median_margin=2):
    # aligned_images = align_images(exposure_images, median_margin)
    # return hdr_poly(exposure_images, exposure_times)

    # debe = cv2.createMergeDebevec()
    # return debe.process(exposure_images, exposure_times)

    my_hdr = hdr_debevec(exposure_images, exposure_times)
    return my_hdr

def tone_map_basic(hdr_image, a=np.array([1, 1, 1])):
    output = np.empty_like(hdr_image)
    total = np.sum(np.log(hdr_image + np.finfo(np.float32).eps)) / hdr_image.size
    Lw_bar = np.exp(total)
    coef = a / Lw_bar
    # print(hdr_image)
    for c in range(output.shape[2]):
        output[:, :, c] = coef[c] * hdr_image[:, :, c]
        output[:, :, c] = output[:, :, c] / (1 + output[:, :, c])

    return output

def tone_mapping(hdr, gamma, l, c):
    tonemap = cv2.createTonemapReinhard(gamma=gamma, light_adapt=l, color_adapt=c)
    ldr = tonemap.process(hdr.copy())
    return np.clip(ldr * 255, 0, 255).astype('uint8')