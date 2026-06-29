RAW_ADC_PATH: str = 'C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\adc_data.bin'

CONST_SPEED_LIGHT = 299_792_458

N_CHIRPS_LOOPS: int = 255
RX_CHANNELS: int = 4
TX_CHANNELS: int = 2
N_FRAMES: int = 10
WAVELENGTH: float = 5e-3
CHIRP_CYCLE_TIME: float = 58.93e-6 + 7e-6
INTER_TX_DELAY: float = 14.73e-6

SLOPE_RATE: float = 67.833e12
ADC_SAMPLES: int = 128
SAMPLING_RATE_HZ: float = 2.5e6

BANDWIDTH: float = SLOPE_RATE * (ADC_SAMPLES / SAMPLING_RATE_HZ)

RANGE_RESOLUTION: float = CONST_SPEED_LIGHT / (2 * BANDWIDTH)
RANGE_MAX: float = CONST_SPEED_LIGHT * ADC_SAMPLES / (2 * BANDWIDTH)
VELOCITY_RESOLUTION: float = WAVELENGTH / (2 * TX_CHANNELS * N_CHIRPS_LOOPS * CHIRP_CYCLE_TIME)
VELOCITY_MAX: float = WAVELENGTH / (4 * TX_CHANNELS * CHIRP_CYCLE_TIME)

if __name__ == '__main__':
    print("Bandwidth Ghz:", BANDWIDTH / 1e6)
    print("Range resolution:", RANGE_RESOLUTION)
    print("Range max:", RANGE_MAX)
    print("Velocity resolution:", VELOCITY_RESOLUTION)
    print("Velocity max:", VELOCITY_MAX)
