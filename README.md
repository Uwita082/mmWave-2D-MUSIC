# IWR6843ISK Point Cloud via 2D-MUSIC

Super-resolution point cloud pipeline for the **Texas Instruments IWR6843ISK** 60 GHz mmWave radar. Raw ADC data is captured with the **DCA1000EVM** evaluation board using **mmWave Studio**, then processed entirely in Python through range FFT, Doppler FFT, CFAR detection, and the MUSIC algorithm to produce high-angular-resolution point clouds.

Two processing variants are provided:

| Mode | Description |
|------|-------------|
| **Azimuth-only** (`PointCloud1DMUSIC`) | 1D MUSIC on the 8-element azimuth virtual array → 2D (x, y) top-down point cloud |
| **Azimuth + Elevation** (`PointCloud2DMUSIC`) | 1D MUSIC for azimuth + monopulse from the elevation row → full 3D (x, y, z) point cloud |

---

## Hardware & Measurement Setup

| Item | Details |
|------|---------|
| Sensor | IWR6843ISK (60 GHz, 3 TX / 4 RX) |
| Data capture board | DCA1000EVM (raw ADC over LVDS) |
| Configuration software | mmWave Studio v2.1.1.0 |
| Output file | `adc_data.bin` (int16, 2's-complement I/Q, non-interleaved) |

### Chirp & Frame Configuration (defaults in `utils/config.py`)

| Parameter                | Value         |
|--------------------------|---------------|
| TX channels (TDM-MIMO)   | 2 or 3        |
| RX channels              | 4             |
| ADC samples per chirp    | 128           |
| Chirps per frame (loops) | 255           |
| Frames                   | 10            |
| Sample rate              | 2500 ksps     |
| Slope rate               | 67.833 MHz/µs |
| Chirp cycle time         | 65.93 µs      |
| Inter-TX delay (TDM)     | 14.73 µs      |
| Wavelength               | 5 mm (60 GHz) |

Derived radar performance:

| Quantity                 | Value             |
|--------------------------|-------------------|
| Bandwidth                | ~3.48 GHz         |
| Range resolution         | ~4.3 cm           |
| Max unambiguous range    | ~5.5 m            |
| Velocity resolution      | ~0.048 m/s (2 TX) |
| Max unambiguous velocity | ±6.1 m/s (2 TX)   |

---

## Virtual Array Layout (IWR6843ISK, 2 TX TDM-MIMO)

```
TX0 → Physical TX1  (azimuth)
TX1 → Physical TX3  (azimuth)

Virtual element indices and positions (d = λ/2 spacing):

 Azimuth row  (y = 0):   v[0..7]   at x = 0,1,2,3,4,5,6,7
```

MUSIC operates on the 8-element azimuth row

## Virtual Array Layout (IWR6843ISK, 3 TX TDM-MIMO)

```
TX0 → Physical TX1  (azimuth)
TX1 → Physical TX2  (elevation)
TX2 → Physical TX3  (azimuth)

Virtual element indices and positions (d = λ/2 spacing):

 Azimuth row  (y = 0):   v[0..7]   at x = 0,1,2,3,4,5,6,7
 Elevation row (y = 1):  v[8..11]  at x = 2,3,4,5
```

MUSIC operates on the 8-element azimuth row; monopulse elevation uses the phase difference between the two rows at the known azimuth.

**Note:** In the case of the chirp configuration for the 3 TX TDM-MIMO, in **mmWave Studio**, we want the first chirp to emit on TX0, second chirp on TX2, and the last chirp on the TX1. 
In that way, we keep for the virtual antenna the first 8 element to be part of the azimuth row, and the last 4 elements are part of the elevation row.

---

## Repository Structure

```
.
├── cfar/
│   ├── __init__.py
│   └── cfar_2d.py                      # CA-CFAR and OS-CFAR detectors
│
├── parser/
│   ├── __init__.py
│   └── parser_dca1000.py               # Raw .bin → [frames, chirps, rx_virtual, samples]
│
├── plots/
│   ├── __init__.py
│   ├── azimuth_spectrum.py             # Per-range azimuth MUSIC spectrum
│   ├── cfar_detections.py              # CFAR detection overlay
│   ├── music_spectrum.py               # Range-azimuth MUSIC heatmap
│   ├── point_cloud.py                  # Interactive 2D / 3D point cloud (Plotly)
│   ├── range.py                        # Range FFT magnitude plot
│   ├── range_azimuth.py                # Range-azimuth map
│   └── range_doppler.py                # Range-Doppler map
│
├── point_cloud/
│   ├── __init__.py
│   ├── MUSIC_azimuth.py                # PointCloud1DMUSIC  (azimuth only)
│   └── MUSIC_azimuth_elevation.py      # PointCloud2DMUSIC  (azimuth + elevation)
│
├── utils/
│   ├── __init__.py
│   └── config.py                       # All radar parameters and derived constants
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Processing Pipeline

```
adc_data.bin
     │
     ▼
[parser_dca1000]
  parse_mmwave_studio_adc()
  LVDS 2I-2Q deinterleave → complex64
  Output: [frames, chirps, rx_virtual, adc_samples]
     │
     ▼
[Range FFT]  (Hanning window + FFT along adc_samples axis)
  Output: [frames, chirps, rx_virtual, range_bins]
     │
     ├── (optional) Static clutter removal
     │   Mean across chirps subtracted → removes zero-Doppler static returns
     │
     ▼
[Doppler FFT]  (Hanning window + FFT along chirps axis + fftshift)
  Output: [frames, doppler_bins, rx_virtual, range_bins]
     │
     ├── (optional) TDM-MIMO phase correction
     │   Corrects inter-TX phase shift induced by Doppler during TDM switching:
     │     TX1 virtual channels (v4–v7):  × exp(−j 2π f_d T_chirp)
     │     TX2 virtual channels (v8–v11): × exp(−j 2π f_d 2 T_chirp)
     │
     ▼
[CFAR Detection]  on range-Doppler power map [frames, doppler, range]
  OS-CFAR (default) or CA-CFAR, 2D sliding window
  + (optional) Peak grouping (local maximum filter) to suppress clustered detections
     │
     ▼
[MUSIC per detection]  for each (range, Doppler) cell that passes CFAR:
  1. Extract virtual array snapshot x[12] (or x[8] azimuth-only)
  2. Form rank-1 covariance R = x xᴴ
  3. Forward-backward spatial smoothing → full-rank Rsmooth (subarray size L=7)
  4. SVD → noise subspace
  5. Scan steering vectors → MUSIC pseudo-spectrum P(θ)
  6. Peak detection with prominence threshold → azimuth estimate(s)
  7. (2D mode) Monopulse elevation from 2-row phase difference
     │
     ▼
[Point cloud output]
  Azimuth-only:  (range_m, velocity_m/s, azimuth_deg, power_dB)
  Azimuth+Elev:  (x, y, z, range_m, velocity_m/s, az_deg, el_deg, power_dB)
```

---

## Key Modules

### `parser/parser_dca1000.py`

Parses the raw binary produced by the DCA1000EVM when capturing IWR6843 data in non-interleaved LVDS mode.

- Reads int16 2's-complement samples directly (no 2¹⁵ offset correction needed)
- Deinterleaves the 2I-2Q LVDS packing: `LVDS[2k] = adc[4k] + j·adc[4k+2]`
- Reshapes to `[frames, chirps, tx, rx, adc_samples]` then collapses TX/RX into a virtual axis `v = tx·N_rx + rx`

### `point_cloud/MUSIC_azimuth.py` - `PointCloud1DMUSIC`

Azimuth-only mode using 2 TX (8 virtual elements). Phase correction compensates the Doppler-induced phase on the second TX's virtual elements.

### `point_cloud/MUSIC_azimuth_elevation.py` - `PointCloud2DMUSIC`

Full 3D mode using 3 TX (12 virtual elements: 8 azimuth + 4 elevation). Azimuth is resolved by 1D MUSIC on the 8-element ULA row; elevation is resolved by monopulse from the phase difference between the two rows after steering out the azimuth component.

### `cfar/cfar_2d.py`

Two CFAR variants operating on the 2D range-Doppler map:

- **CA-CFAR** (`ca_cfar_2d`): cell-averaging, fast convolution-based implementation
- **OS-CFAR** (`os_cfar_2d`): ordered-statistics, robust to contaminating targets; threshold multiplier α solved by Newton's method for exact PFA

Both are wrapped in frame-batched versions (`cfar_on_cube`, `os_cfar_on_cube`) and an optional `peak_group` maximum-filter NMS step.

---

## Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Capture data

Configure the IWR6843ISK in **mmWave Studio**:

1. Set the chirp profile (slope, ADC samples, idle/ramp times) to match `utils/config.py`
2. Enable TDM-MIMO (2 TX for azimuth-only, 3 TX for azimuth+elevation)
3. Start the DCA1000EVM capture and run the measurement
4. Point the output file of the mmWave Studio post-processing `adc_data.bin` to the path set in `config.py` (`RAW_ADC_PATH`)

### 3. Run azimuth-only point cloud (2 TX)

```python
from parser.parser_dca1000 import parse_mmwave_studio_adc
from point_cloud.MUSIC_azimuth import PointCloud1DMUSIC
from utils.config import RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, N_FRAMES

adc_data = parse_mmwave_studio_adc(RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, tx=2, num_frames=N_FRAMES)

p = PointCloud1DMUSIC()
p.run(adc_data, static_removal=True, phase_shift=True)
```

### 4. Run azimuth + elevation point cloud (3 TX)

```python
from parser.parser_dca1000 import parse_mmwave_studio_adc
from point_cloud.MUSIC_azimuth_elevation import PointCloud2DMUSIC
from utils.config import RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, N_FRAMES

adc_data = parse_mmwave_studio_adc(RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, tx=3, num_frames=N_FRAMES)

p = PointCloud2DMUSIC()
p.run(adc_data, static_removal=True, phase_shift=True)
```

**`static_removal`**: subtracts the mean across chirps to cancel stationary clutter (walls, furniture). Disable when you want to detect static targets such as a stationary person.

**`phase_shift`**: applies the TDM-MIMO Doppler phase correction before MUSIC. Should be enabled whenever targets are moving.

---

## Configuration Reference (`utils/config.py`)

| Variable | Description |
|----------|-------------|
| `RAW_ADC_PATH` | Path to the `.bin` file produced by mmWave Studio |
| `N_CHIRPS_LOOPS` | Chirps per frame (determines Doppler resolution) |
| `RX_CHANNELS` | Physical RX antennas (4 on IWR6843ISK) |
| `TX_CHANNELS` | Active TX antennas in TDM-MIMO (2 or 3) |
| `N_FRAMES` | Number of frames to process |
| `ADC_SAMPLES` | ADC samples per chirp (determines range resolution) |
| `SAMPLING_RATE_HZ` | ADC sampling rate |
| `SLOPE_RATE` | FMCW chirp slope (Hz/s) |
| `CHIRP_CYCLE_TIME` | Total chirp cycle duration (ramp + idle) |
| `INTER_TX_DELAY` | Delay between TDM TX firings |
| `WAVELENGTH` | Carrier wavelength (m) |

---

## Dependencies

Python **3.12** is required.

```
scipy==1.18.0
numpy==2.5.0
plotly==6.8.0
```

Install with:

```bash
pip install -r requirements.txt
```

`plotly` is used for interactive visualizations.

---

## Algorithm Notes

### Why MUSIC instead of FFT beamforming?

The IWR6843ISK virtual array has only 8 azimuth elements, giving a conventional FFT angular resolution of ~14° at broadside. MUSIC is a subspace method that resolves targets below the Rayleigh limit, achieving <1° effective resolution when the signal-to-noise ratio is sufficient.

### Forward-Backward Spatial Smoothing

A single chirp snapshot produces a rank-1 covariance matrix, which collapses the noise subspace and makes MUSIC singular. Forward-backward spatial smoothing on a subarray of length L=7 decorrelates the signals and restores full rank, at the cost of reducing the effective aperture from 8 to 7 elements.

### TDM-MIMO Phase Correction

In time-division multiplexed MIMO, each TX fires in a different time slot within the same chirp cycle. A moving target accumulates additional phase between TX firings proportional to its radial velocity and the inter-TX delay. Without correction, this phase error appears as a fictitious angle offset in the virtual array steering vector. The correction applied here is:

```
φ_correction(TX_k) = −2π · f_Doppler · k · T_chirp
```

where `f_Doppler` is the Doppler frequency of the bin under test and `k` is the TX index (0-based).

### Static Clutter Removal

Subtracting the mean across the slow-time (chirp) axis removes any return whose phase is constant across chirps (i.e., static scatterers). This is equivalent to a DC-block filter in the Doppler domain. It is effective for removing wall and ground clutter but will also remove or attenuate very slow-moving targets (< velocity resolution).
