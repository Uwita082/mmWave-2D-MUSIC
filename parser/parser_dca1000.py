import numpy as np

from utils.config import RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, TX_CHANNELS, N_FRAMES


def parse_mmwave_studio_adc(filename, num_adc_samples, num_chirps, num_rx, num_tx, num_frames):
    """
    Parse raw ADC .bin from mmWave Studio / DCA1000 for IWR6843 (xWR16xx),
    complex, 4 RX, 2 LVDS lanes, NON-INTERLEAVED (Section 24.8).

    Per-RX layout in the stream is [I, I, Q, Q] groups:
        LVDS[2k]   = adc[4k]   + j*adc[4k+2]
        LVDS[2k+1] = adc[4k+1] + j*adc[4k+3]
    Within each chirp: all of RX0's N samples, then RX1, RX2, RX3.
    DCA1000 stores int16 2's-complement (no 2^15 offset correction).

    Returns: [frames, chirps(loops), virtual_rx, adc_samples], complex64,
             with virtual index v = tx*num_rx + rx (RX-fastest within each TX).
    """
    adc_data = np.fromfile(filename, dtype=np.int16)

    num_samples_per_chirp = num_adc_samples * num_rx * 2
    expected = num_frames * num_chirps * num_tx * num_samples_per_chirp
    assert expected == len(adc_data), f"{expected} != {len(adc_data)}"

    # --- 2I-2Q deinterleave -> complex, length = len/2 ---
    g = adc_data.reshape(-1, 4)
    lvds = np.empty((g.shape[0], 2), dtype=np.complex64)
    lvds[:, 0] = g[:, 0] + 1j * g[:, 2]
    lvds[:, 1] = g[:, 1] + 1j * g[:, 3]
    lvds = lvds.reshape(-1)

    # Each chirp = [RX0(N samples), RX1(N), RX2(N), RX3(N)], chirps TX-fastest
    total_chirps = num_frames * num_chirps * num_tx
    cube = lvds.reshape(total_chirps, num_rx, num_adc_samples)
    # -> [frames, loops, tx, rx, adc_samples]
    cube = cube.reshape(num_frames, num_chirps, num_tx, num_rx, num_adc_samples)

    # Collapse (tx, rx) into virtual axis v = tx*num_rx + rx
    num_virtual_rx = num_tx * num_rx
    cube = np.ascontiguousarray(cube)
    return cube.reshape(num_frames, num_chirps, num_virtual_rx, num_adc_samples)


if __name__ == '__main__':
    adc_data = parse_mmwave_studio_adc(RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, TX_CHANNELS, N_FRAMES)

    # frames, chirps(loops), virtual_rx, adc_samples
    print(np.shape(adc_data))
