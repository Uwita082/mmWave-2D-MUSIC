import numpy as np
import plotly.graph_objects as go

_COLORSCALE = [
    [0.00, '#000080'],
    [0.20, '#0000ff'],
    [0.40, '#00ffff'],
    [0.60, '#00ff00'],
    [0.80, '#ffff00'],
    [1.00, '#ff0000'],
]

_LAYOUT_BASE = dict(
    template='plotly_dark',
    paper_bgcolor='#0d0d0d',
    height=700,
    width=850,
)


def _to_db(x: np.ndarray, clip_db: float) -> np.ndarray:
    x_db = 10 * np.log10(x / x.max() + 1e-12)
    return np.clip(x_db, -clip_db, 0)


def plot_azimuth_elevation_3d(
    spectrum: np.ndarray,
    range_idx: float,
    azimuth_fov_degrees: float = 60,
    elevation_fov_degrees: float = 30,
    db_range: float = 30,
) -> None:
    """
    3D surface plot of the MUSIC pseudo-spectrum for a single range bin.

    Parameters
    ----------
    spectrum : ndarray, shape [n_az, n_el]
        Raw MUSIC pseudo-spectrum slice at one range index (not in dB).
    range_idx : int
        Range bin index shown in the plot title.
    azimuth_fov_degrees : half-FOV used when scanning azimuth.
    elevation_fov_degrees : half-FOV used when scanning elevation.
    db_range : colour dynamic range in dB (0 dB = peak).
    """
    n_az, n_el = spectrum.shape
    az_deg = np.linspace(-azimuth_fov_degrees, azimuth_fov_degrees, n_az)
    el_deg = np.linspace(-elevation_fov_degrees, elevation_fov_degrees, n_el)

    power_db = _to_db(spectrum, db_range)   # [n_az, n_el]

    AZ, EL = np.meshgrid(az_deg, el_deg, indexing='ij')

    fig = go.Figure(go.Surface(
        x=AZ,
        y=EL,
        z=power_db,
        colorscale=_COLORSCALE,
        cmin=-db_range,
        cmax=0,
        colorbar=dict(
            thickness=14,
            title=dict(text='dB', side='right'),
            tickfont=dict(size=11),
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f'MUSIC Spectrum — Range {range_idx}',
            x=0.5,
            font=dict(size=15),
        ),
        scene=dict(
            xaxis=dict(title='Azimuth (degrees)', gridcolor='rgba(255,255,255,0.08)'),
            yaxis=dict(title='Elevation (degrees)', gridcolor='rgba(255,255,255,0.08)'),
            zaxis=dict(title='Power (dB)', gridcolor='rgba(255,255,255,0.08)'),
        ),
        **_LAYOUT_BASE,
    )

    fig.show()
