from parser.parser_dca1000 import parse_mmwave_studio_adc
from point_cloud.MUSIC_azimuth_elevation import PointCloud2DMUSIC
from utils.config import RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, N_FRAMES

if __name__ == '__main__':
    adc_data = parse_mmwave_studio_adc(RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, 3, N_FRAMES)

    p = PointCloud2DMUSIC()
    p.run(adc_data, static_removal=False, phase_shift=True)


