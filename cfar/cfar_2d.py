"""
2D CA-CFAR detector for mmWave radar range-azimuth maps.

Input cube has shape [frames, range, azimuth]. For each frame we estimate the
local noise around every cell from a ring of training cells (skipping a guard
region around the cell under test), then declare a detection if the cell's
power exceeds an adaptive threshold tied to the chosen P_fa.
"""

import numpy as np
from scipy.ndimage import generic_filter
from scipy.signal import convolve2d


def ca_cfar_2d(power_map,
               guard_range=2, guard_az=2,
               train_range=4, train_az=4,
               pfa=1e-4):
    """
    Cell-Averaging CFAR on a 2D range-azimuth power map.

    Parameters
    ----------
    power_map : (n_range, n_az) ndarray of power values (|x|^2).
                NOT dB, NOT complex, NOT magnitude.
    guard_range, guard_az : half-width of the guard region (cells per side).
    train_range, train_az : half-width of the training region (cells per side
                            beyond the guards).
    pfa : desired probability of false alarm.

    Returns
    -------
    detections : (n_range, n_az) bool ndarray
    threshold  : (n_range, n_az) float ndarray (useful for debugging)
    """
    win_r = guard_range + train_range
    win_a = guard_az + train_az

    # Kernel covering the full window: training + guard + CUT
    full_kernel = np.ones((2 * win_r + 1, 2 * win_a + 1), dtype=np.float64)

    # Kernel covering only the guard region + CUT (the part we subtract out)
    guard_kernel = np.zeros_like(full_kernel)
    guard_kernel[train_range:train_range + 2 * guard_range + 1,
                 train_az:train_az + 2 * guard_az + 1] = 1.0

    # Training-cell mask. Convolving the power map with this gives
    # the sum of training-cell powers at every position.
    train_kernel = full_kernel - guard_kernel
    n_train = int(train_kernel.sum())

    train_sum = convolve2d(power_map, train_kernel,
                           mode='same', boundary='symm')
    noise_est = train_sum / n_train

    # CA-CFAR scaling for exponentially distributed noise power
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    threshold = alpha * noise_est

    detections = power_map > threshold
    return detections, threshold


def ca_cfar_on_cube(data_cube,
                 guard_range=2, guard_az=2,
                 train_range=8, train_az=8,
                 pfa=1e-5):
    """
    Apply 2D CA-CFAR to each frame of a [frames, range, azimuth] cube.

    Accepts complex IQ or already-real input. Complex is squared to power;
    real input is treated as power directly (if yours is magnitude, square
    it before calling).
    """
    if np.iscomplexobj(data_cube):
        power = data_cube.real ** 2 + data_cube.imag ** 2
    else:
        power = data_cube.astype(np.float64)

    detections = np.zeros(power.shape, dtype=bool)
    thresholds = np.zeros(power.shape, dtype=np.float64)
    for i in range(power.shape[0]):
        detections[i], thresholds[i] = ca_cfar_2d(
            power[i],
            guard_range=guard_range, guard_az=guard_az,
            train_range=train_range, train_az=train_az,
            pfa=pfa,
        )
    return detections, thresholds

def os_cfar_2d(power_map,
               guard_doppler=2, guard_range=2,
               train_doppler=8, train_range=8,
               k_rank=None, pfa=1e-5):
    """
    Ordered-Statistics CFAR on a 2D power map.

    For each cell under test, ranks the training cells and uses the k-th
    smallest as the noise estimate. This makes the estimator robust to
    a minority of contaminating bright cells (multipath trails, secondary
    targets, clutter ridges) that would bias CA-CFAR's mean.

    Parameters
    ----------
    power_map : (n_range, n_az) ndarray of LINEAR power values (|x|^2).
    guard_range, guard_az : half-width of guard region.
    train_range, train_az : half-width of training region beyond guards.
    k_rank : int, the rank to use (1-indexed). Default is 0.75 * n_train,
             which gives good robustness to ~25% contamination. Lower
             k_rank => more aggressive bright-cell rejection but higher
             noise variance.
    pfa : desired probability of false alarm.

    Returns
    -------
    detections : (n_range, n_az) bool ndarray
    threshold  : (n_range, n_az) float ndarray
    """
    win_r = guard_doppler + train_doppler
    win_a = guard_range + train_range
    H, W = 2 * win_r + 1, 2 * win_a + 1

    # Boolean mask for training cells: 1 in training ring, 0 in guard+CUT
    train_mask = np.ones((H, W), dtype=bool)
    train_mask[train_doppler:train_doppler + 2 * guard_doppler + 1,
               train_range:train_range + 2 * guard_range + 1] = False
    n_train = int(train_mask.sum())

    if k_rank is None:
        k_rank = int(0.75 * n_train)
    k_rank = max(1, min(n_train, k_rank))

    # generic_filter feeds a flat 1D array of the window's values; we mask
    # to training cells, sort, and pick the k-th smallest.
    flat_mask = train_mask.ravel()

    def _os_estimate(window_flat):
        train_values = window_flat[flat_mask]
        # np.partition is O(n) for finding the k-th element vs O(n log n) sort
        return np.partition(train_values, k_rank - 1)[k_rank - 1]

    noise_est = generic_filter(power_map, _os_estimate,
                                size=(H, W), mode='reflect')

    # OS-CFAR scaling factor for exponentially distributed noise power.
    # Exact closed form: alpha = product over i=0..k-1 of (n_train - i) /
    #                   (n_train - i - alpha_factor), solved for the alpha
    #                   that gives the target PFA. We use the standard
    #                   approximation that's accurate for typical parameters.
    alpha = _os_alpha(n_train, k_rank, pfa)
    threshold = alpha * noise_est

    detections = power_map > threshold
    return detections, threshold


def _os_alpha(N, k, pfa, tol=1e-10, max_iter=200):
    """
    Compute the OS-CFAR threshold multiplier alpha satisfying the target PFA
    for exponentially distributed noise power.

    PFA = product_{i=0..k-1} (N - i) / (N - i + alpha)

    Solves by Newton's method in log space.
    """
    # Initial guess from CA-CFAR equivalent
    alpha = N * (pfa ** (-1.0 / N) - 1.0)

    for _ in range(max_iter):
        # log(PFA) = sum log((N - i) / (N - i + alpha))
        idx = np.arange(k)
        denom = (N - idx) + alpha
        log_pfa = np.sum(np.log((N - idx) / denom))
        f = log_pfa - np.log(pfa)
        # d/d_alpha of log_pfa = sum -1 / (N - i + alpha)
        df = -np.sum(1.0 / denom)
        step = f / df
        alpha -= step
        if abs(step) < tol * alpha:
            break

    return alpha


def os_cfar_on_cube(data_cube,
                    guard_doppler=2, guard_range=2,
                    train_doppler=8, train_range=8,
                    k_rank=None, pfa=1e-5):
    """
    Apply 2D OS-CFAR to each frame of a [frames, doppler, range] cube.
    Mirrors the CA-CFAR signature so it's a drop-in replacement.
    """
    if np.iscomplexobj(data_cube):
        power = data_cube.real ** 2 + data_cube.imag ** 2
    else:
        power = data_cube.astype(np.float64)

    detections = np.zeros(power.shape, dtype=bool)
    thresholds = np.zeros(power.shape, dtype=np.float64)
    for i in range(power.shape[0]):
        detections[i], thresholds[i] = os_cfar_2d(
            power[i], guard_doppler, guard_range, train_doppler, train_range, k_rank, pfa
        )
    return detections, thresholds


# ---------------------------------------------------------------------------
# Optional: simple peak grouping. CFAR detections cluster around strong
# targets; this collapses each cluster to its local maximum.
# ---------------------------------------------------------------------------
def peak_group(power_map, detections, neighborhood=3):
    """
    Keep only detections that are a local maximum within a (2n+1) x (2n+1)
    neighborhood. Cheap stand-in for NMS / DBSCAN.
    """
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(power_map, size=2 * neighborhood + 1)
    return detections & (power_map == local_max)


if __name__ == "__main__":
    # Quick synthetic check: exponential noise plus a couple of planted targets.
    rng = np.random.default_rng(0)
    frames, n_range, n_az = 4, 128, 64
    cube = rng.exponential(scale=1.0, size=(frames, n_range, n_az))
    cube[:, 40, 20] += 50.0
    cube[:, 80, 45] += 80.0

    print(cube[0])

    det, thr = ca_cfar_on_cube(cube,
                            guard_range=2, guard_az=2,
                            train_range=4, train_az=4,
                            pfa=1e-4)
    print("Raw CFAR detections per frame:", det.reshape(frames, -1).sum(axis=1))

    grouped = np.array([peak_group(cube[i], det[i]) for i in range(frames)])
    print("After peak grouping:           ", grouped.reshape(frames, -1).sum(axis=1))
    # Expect ~2 detections per frame after grouping.