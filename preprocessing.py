"""=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                   Adaptive Mean Phase Filtering
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="""

import cv2
import numpy as np


def adaptive_mean_phase_filter(img, kernel_size=5, phase_weight=0.3):

    img = img.astype(np.float32)

    # Step 1 : Adaptive Mean Filtering
    local_mean = cv2.blur(img, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(img**2,(kernel_size, kernel_size))
    local_var = local_sq_mean - local_mean**2
    noise_var = np.mean(local_var)

    adaptive = local_mean + (
        np.maximum(local_var - noise_var, 0)
        / (local_var + 1e-8)) * (img - local_mean)

    # Step 2 : Phase Preservation
    fft = np.fft.fft2(adaptive)
    magnitude = np.abs(fft)
    phase = np.angle(fft)

    magnitude = cv2.GaussianBlur(magnitude, (5, 5), 0)
    reconstructed = magnitude * np.exp(1j * phase)
    phase_img = np.real(np.fft.ifft2(reconstructed))
    
    # Step 3 : Combined
    filtered = ((1 - phase_weight) * adaptive + phase_weight * phase_img)
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)

    return filtered.astype(np.uint8)
