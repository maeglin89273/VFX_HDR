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
        print_alignment_result(x, y)
        result.append(shift_color_image(images[i], x, y) if not is_big_shift(x, y) else images[i])

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
    for ix, iy in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]: #from inner to outter
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

BIG_SHIFT_THRESHOLD = 50
def is_big_shift(x, y):
    return np.abs(x) > BIG_SHIFT_THRESHOLD or np.abs(y) > BIG_SHIFT_THRESHOLD

def print_alignment_result(x, y):
    print('alignment shift: %d, %d' % (x, y))
    if is_big_shift(x, y):
        print('WARNING: This alignment has big shifts. We skip alignment on this image')


def compute_bitmap(image, median_margin):
    median = np.median(image)
    result = image > median
    exclusive_bitmap = (image > median + median_margin) | (image < median - median_margin)
    return result, exclusive_bitmap



# Implementation of Debevec HDR

COLOR_N = 256
Z_MID = 127
# WEIGHT_Z = np.array([z + 1 if z < Z_MID else COLOR_N - z + 1 for z in range(COLOR_N)])
WEIGHT_Z = np.array([z + 1 if z < Z_MID else Z_MID for z in range(COLOR_N)])
# WEIGHT_Z = np.array([Z_MID for z in range(COLOR_N)])
WEIGHT_Z = WEIGHT_Z / np.sum(WEIGHT_Z)

def hdr_debevec(exposure_images, exposure_times, l=8):
    exposure_images = np.array(exposure_images)
    sampling_area = compute_good_sampling_area(exposure_images[exposure_images.shape[0] // 2])
    channel_images = [exposure_images[:, :, :, c] for c in range(exposure_images.shape[-1])]

    ln_dt = np.log(exposure_times)

    g_funcs = []
    for c, images in enumerate(channel_images):
        Z = reshape_to_z_and_sample(images, sampling_area)
        A = np.zeros((Z.size + 1 + COLOR_N - 2, COLOR_N + Z.shape[0]))
        b = np.zeros(A.shape[0])

        w_z_ij = WEIGHT_Z[Z].ravel()
        z_ij = Z.ravel()
        row_indices = np.arange(z_ij.size)

        A[row_indices, z_ij] = w_z_ij
        A[row_indices, COLOR_N + np.repeat(np.arange(Z.shape[0]), Z.shape[1])] = -w_z_ij
        b[:z_ij.size] = w_z_ij * np.tile(ln_dt, Z.shape[0])

        mid_idx = z_ij.size
        A[mid_idx, 127] = 1
        b[mid_idx] = 5.5 #heuristic value

        row_indices = np.arange(mid_idx + 1, A.shape[0])
        z_0_i = np.arange(0, row_indices.size)
        z_1_i = 1 + z_0_i
        z_2_i = 2 + z_0_i
        A[row_indices, z_0_i] = l * WEIGHT_Z[z_0_i]
        A[row_indices, z_1_i] = -2 * l * WEIGHT_Z[z_1_i]
        A[row_indices, z_2_i] = l * WEIGHT_Z[z_2_i]

        vars, error, _, __ = np.linalg.lstsq(A, b)
        g_funcs.append(vars[:COLOR_N])

    utils.show_crf_curves(g_funcs)
    return reconstruct_hdr(exposure_images, g_funcs, ln_dt)


def compute_good_sampling_area(median_exposure_image):
    return cv2.Canny(cv2.cvtColor(median_exposure_image, cv2.COLOR_BGR2GRAY), 150, 200) < 127

def reshape_to_z_and_sample(images, sampling_area, samples=1000):
    good_sampling_positions = np.where(sampling_area.ravel())[0]
    sampling_positions = good_sampling_positions[np.random.choice(good_sampling_positions.size, samples)]
    images = images.reshape((images.shape[0], -1)).T
    return images[sampling_positions]


#Implementation of polynomial HDR
def hdr_poly(exposure_images, exposure_times, max_order=25):
    exposure_images = np.array(exposure_images)
    sampling_area = compute_good_sampling_area(exposure_images[exposure_images.shape[0] // 2])
    channel_images = [exposure_images[:, :, :, c] for c in range(3)]
    exposure_ratios = compute_exposure_ratios(exposure_times)

    g_funcs = []
    for c, images in enumerate(channel_images):
        g_func = find_poly_g_func(images, sampling_area, exposure_ratios, max_order)
        g_funcs.append(g_func)

    utils.show_crf_curves(g_funcs)
    return reconstruct_hdr(exposure_images, g_funcs, np.log(exposure_times))


def find_poly_g_func(images, sampling_area, exposure_ratios, max_order):
    min_error = np.inf
    best_coefs = None
    Z = reshape_to_z_and_sample(images, sampling_area) / (COLOR_N - 1) # Since the numerical problem, shrink 0-255 to 0~1
    ratio_vector = np.repeat(exposure_ratios, Z.shape[0])
    N = Z.shape[0]

    for poly_n in range(2, max_order + 1):
        Z_ji_m = np.tile(Z.T.reshape((-1, 1)), (1, poly_n + 1)) ** np.arange(poly_n, -1, -1)
        A = Z_ji_m[:-N, :] - (ratio_vector[:, np.newaxis] * Z_ji_m[N:, :])
        A = np.vstack((A, np.ones(A.shape[1])))
        b = np.zeros(A.shape[0])
        b[-1] = 1
        coefs, error, _, __ = np.linalg.lstsq(A, b)
        if error < min_error:
            min_error = error
            best_coefs = coefs

    print('largest polynomial degree: %s' % (best_coefs.size - 1))
    poly = np.poly1d(best_coefs)
    poly_g_func = poly(np.arange(0, COLOR_N) / (COLOR_N - 1))
    return np.log(np.clip(poly_g_func, 0, 1) + np.finfo(np.float32).eps)


def compute_exposure_ratios(exposure_times):
    return np.array([exposure_times[i] / exposure_times[i + 1] for i in range(len(exposure_times) - 1)])

def reconstruct_hdr(exposure_images, g_funcs, ln_dt):
    hdr_image = np.empty(exposure_images.shape[1:])
    channel_images = [exposure_images[:, :, :, c] for c in range(exposure_images.shape[-1])]
    for c, (images, g_func) in enumerate(zip(channel_images, g_funcs)):
        g_func = g_funcs[1]
        Z = images.reshape((images.shape[0], -1)).T

        w_z_ij = WEIGHT_Z[Z]
        ln_E = np.sum((g_func[Z] - np.tile(ln_dt, (Z.shape[0], 1))) * w_z_ij, axis=1) / np.sum(w_z_ij, axis=1)

        hdr_image[:, :, c] = np.exp(ln_E).reshape(hdr_image.shape[:-1])

    return hdr_image

def hdr(exposure_images, exposure_times, alignment=True, algorithm='debevec', median_margin=2):
    if alignment:
        aligned_images = align_images(exposure_images, median_margin)
    else:
        aligned_images = exposure_images

    if algorithm == 'debevec':
        hdr_image = hdr_debevec(aligned_images, exposure_times)
    elif algorithm == 'poly':
        hdr_image = hdr_poly(aligned_images, exposure_times)
    else:
        hdr_image = hdr_debevec(aligned_images, exposure_times)

    return hdr_image.astype('float32') #hdr file works with float32

