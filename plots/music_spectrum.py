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
    """
    Maps (range, angle) data onto a Cartesian grid for a fan-shaped heatmap.
    x = r·sin(θ)  (cross-range), y = r·cos(θ)  (down-range).
    NaN outside the sector keeps the background dark.
    """
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


def _make_sector_figure(x, y, z, title, db_range, frame_idx, anchor,
                        xlabel='Cross-range (meters)', ylabel='Down-range (meters)'):
    cb = dict(thickness=14, title=dict(text='dB', side='right'), tickfont=dict(size=11))
    fig = go.Figure(go.Heatmap(
        x=x, y=y, z=z,
        colorscale=_COLORSCALE, zmin=-db_range, zmax=0,
        colorbar=cb,
        hoverongaps=False,
    ))
    fig.update_xaxes(title_text=xlabel, **_AXIS_STYLE)
    fig.update_yaxes(title_text=ylabel,
                     scaleanchor=anchor, scaleratio=1, **_AXIS_STYLE)
    fig.update_layout(
        title=dict(text=f'{title} — frame {frame_idx}', x=0.5, font=dict(size=15)),
        **_LAYOUT_BASE,
    )
    return fig


def plot_music_spectrum(p_music: np.ndarray,
                        azimuth_fov_degrees: float = 60,
                        elevation_fov_degrees: float = 15,
                        frame_idx: int = 0,
                        db_range: float = 80,
                        grid_res: int = 350,
                        sensor_height: float = 1.4) -> None:
    """
    Two separate polar-sector figures: Range–Azimuth and Range–Elevation.

    Parameters
    ----------
    p_music : ndarray, shape [frames, range, n_az, n_el]
        Raw MUSIC pseudo-spectrum (not in dB).
    azimuth_fov_degrees : half-FOV used when scanning azimuth angles.
    elevation_fov_degrees : half-FOV used when scanning elevation angles.
    frame_idx : which frame to display.
    db_range : colour dynamic range in dB (0 dB = peak).
    grid_res : resolution of the interpolated Cartesian grid (pixels per side).
    sensor_height : height of sensor above ground [m]; shifts the elevation
                    axis so that 0 = ground and sensor sits at sensor_height.
    """
    spectrum = p_music[frame_idx]           # [range, n_az, n_el]
    n_range, n_az, n_el = spectrum.shape
    range_bins = np.arange(n_range, dtype=float) * RANGE_RESOLUTION

    az_deg = np.linspace(-azimuth_fov_degrees, azimuth_fov_degrees, n_az)
    el_deg = np.linspace(-elevation_fov_degrees, elevation_fov_degrees, n_el)

    range_az_db = _to_db(spectrum.max(axis=2), db_range)   # [range, n_az]
    range_el_db = _to_db(spectrum.max(axis=1), db_range)   # [range, n_el]

    x_az, y_az, z_az = _polar_sector_grid(az_deg, range_bins, range_az_db, grid_res)
    x_el, y_el, z_el = _polar_sector_grid(el_deg, range_bins, range_el_db, grid_res)

    # x_el is R·sin(el) — height relative to sensor; shift to absolute ground height
    x_el = x_el + sensor_height

    fig_az = _make_sector_figure(x_az, y_az, z_az,
                                  'Range–Azimuth MUSIC', db_range, frame_idx, 'x')
    fig_el = _make_sector_figure(y_el, x_el, z_el.T,
                                  'Range–Elevation MUSIC', db_range, frame_idx, 'x',
                                  xlabel='Down-range (meters)',
                                  ylabel='Height (meters)')

    fig_az.show()
    fig_el.show()


def plot_music_range_azimuth_heatmap(p_music_2d: np.ndarray,
                                     azimuth_fov_degrees: float = 60,
                                     frame_idx: int = 0,
                                     db_range: float = 80,
                                     grid_res: int = 350) -> None:
    """
    Polar-sector Range–Azimuth heatmap of the 1-D MUSIC pseudo-spectrum.

    Parameters
    ----------
    p_music_2d : ndarray, shape [range_bins, n_az]
        Raw MUSIC pseudo-spectrum (not in dB).
    azimuth_fov_degrees : half-FOV used when scanning azimuth angles.
    frame_idx : which frame index (used for title only).
    db_range : colour dynamic range in dB (0 dB = peak).
    grid_res : resolution of the interpolated Cartesian grid.
    """
    n_range, n_az = p_music_2d.shape
    range_bins = np.arange(n_range, dtype=float) * RANGE_RESOLUTION
    az_deg = np.linspace(-azimuth_fov_degrees, azimuth_fov_degrees, n_az)

    p_db = _to_db(p_music_2d, db_range)  # [range, n_az]

    x, y, z = _polar_sector_grid(az_deg, range_bins, p_db, grid_res)
    fig = _make_sector_figure(x, y, z, 'Range–Azimuth MUSIC', db_range, frame_idx, 'x')
    fig.show()
