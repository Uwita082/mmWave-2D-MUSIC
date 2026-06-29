import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

from utils.config import RANGE_RESOLUTION

_COLORSCALE = [
    [0.00, '#000080'],
    [0.20, '#0000ff'],
    [0.40, '#00ffff'],
    [0.60, '#00ff00'],
    [0.80, '#ffff00'],
    [1.00, '#ff0000'],
]

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.08)',
    zeroline=True,
    zerolinecolor='rgba(255,255,255,0.25)',
    tickfont=dict(size=11),
)

_LAYOUT_BASE = dict(
    template='plotly_dark',
    height=680,
    width=780,
    paper_bgcolor='#0d0d0d',
    plot_bgcolor='#0d0d0d',
)


def _to_db(x: np.ndarray, clip_db: float) -> np.ndarray:
    x_db = 10 * np.log10(x / x.max() + 1e-12)
    return np.clip(x_db, -clip_db, 0)


def _polar_sector_grid(angle_deg: np.ndarray, range_bins: np.ndarray,
                        z_2d: np.ndarray, grid_res: int = 350):
    theta = np.deg2rad(angle_deg)
    R, THETA = np.meshgrid(range_bins, theta, indexing='ij')
    X = R * np.sin(THETA)
    Y = R * np.cos(THETA)

    x_lin = np.linspace(X.min(), X.max(), grid_res)
    y_lin = np.linspace(Y.min(), Y.max(), grid_res)
    XI, YI = np.meshgrid(x_lin, y_lin)

    ZI = griddata(
        (X.ravel(), Y.ravel()),
        z_2d.ravel(),
        (XI, YI),
        method='linear',
        fill_value=np.nan,
    )
    return x_lin, y_lin, ZI


def plot_range_azimuth(
    p_music: np.ndarray,
    azimuth_fov_degrees: float = 60,
    frame_idx: int = 0,
    db_range: float = 80,
    grid_res: int = 350,
) -> None:
    """
    Polar-sector Range–Azimuth heatmap for 1-D MUSIC output.

    Parameters
    ----------
    p_music : ndarray, shape [frames, range, n_az]
        Raw MUSIC pseudo-spectrum (not in dB).
    azimuth_fov_degrees : half-FOV used when scanning azimuth.
    frame_idx : which frame to display.
    db_range : colour dynamic range in dB (0 dB = peak).
    grid_res : resolution of the interpolated Cartesian grid.
    """
    spectrum = p_music[frame_idx]           # [range, n_az]
    n_range, n_az = spectrum.shape
    range_bins = np.arange(n_range, dtype=float) * RANGE_RESOLUTION

    az_deg = np.linspace(-azimuth_fov_degrees, azimuth_fov_degrees, n_az)
    range_az_db = _to_db(spectrum, db_range)   # [range, n_az]

    x, y, z = _polar_sector_grid(az_deg, range_bins, range_az_db, grid_res)

    cb = dict(thickness=14, title=dict(text='dB', side='right'), tickfont=dict(size=11))
    fig = go.Figure(go.Heatmap(
        x=x, y=y, z=z,
        colorscale=_COLORSCALE, zmin=-db_range, zmax=0,
        colorbar=cb,
        hoverongaps=False,
    ))
    fig.update_xaxes(title_text='Cross-range (meters)', **_AXIS_STYLE)
    fig.update_yaxes(title_text='Down-range (meters)',
                     scaleanchor='x', scaleratio=1, **_AXIS_STYLE)
    fig.update_layout(
        title=dict(
            text=f'Range–Azimuth MUSIC — frame {frame_idx}',
            x=0.5,
            font=dict(size=15),
        ),
        **_LAYOUT_BASE,
    )
    fig.show()
