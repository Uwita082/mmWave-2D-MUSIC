import numpy as np
import matplotlib.pyplot as plt

from mmwave.dsp.cfar import ca_


def plot_range_fft(range_fft: np.ndarray, ADC_SAMPLES, RANGE_RES) -> None:
    range_axis = np.arange(ADC_SAMPLES) * RANGE_RES

    mag = np.abs(range_fft)
    mag_db = 20 * np.log10(mag / (ADC_SAMPLES * 2 ** 15) + 1e-10)

    threshold_db, noise_floor_db = ca_(mag_db, guard_len=2, noise_len=4, mode='wrap', l_bound=12)
    detections = mag_db > threshold_db

    plt.figure(figsize=(10, 4))
    plt.plot(range_axis, mag_db, label='Range FFT')
    plt.plot(range_axis, threshold_db, color='orange', linestyle='--', label='CA threshold')
    plt.plot(range_axis, noise_floor_db, color='green', linestyle=':', label='Noise floor')
    plt.scatter(range_axis[detections], mag_db[detections], color='red', zorder=5, label='CA detections')
    plt.legend()
    plt.xlabel('Range [m]')
    plt.ylabel('Magnitude [dBFS]')
    plt.title('Range FFT (Chirp 0, RX 0)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()