import numpy as np
import plotly.graph_objects as go

from utils.config import RANGE_RESOLUTION, VELOCITY_MAX, N_CHIRPS_LOOPS

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
    width=820,
    paper_bgcolor='#0d0d0d',
    plot_bgcolor='#0d0d0d',
)


def plot_range_doppler(
    rd_power: np.ndarray,
    frame_idx: int = 0,
    db_range: float = 80,
) -> None:
    """
    Range–Doppler heatmap.

    Parameters
    ----------
    rd_power  : ndarray, shape [frames, doppler_bins, range_bins]
                Linear power (fftshifted along doppler axis).
    frame_idx : which frame to display.
    db_range  : colour dynamic range in dB (0 dB = peak).
    """
    frame = rd_power[frame_idx]  # [doppler_bins, range_bins]
    n_doppler, n_range = frame.shape

    ranges = np.arange(n_range) * RANGE_RESOLUTION
    velocities = np.linspace(-VELOCITY_MAX, VELOCITY_MAX, n_doppler, endpoint=False)

    power_db = 10 * np.log10(frame / frame.max() + 1e-12)
    power_db = np.clip(power_db, -db_range, 0)

    cb = dict(thickness=14, title=dict(text='dB', side='right'), tickfont=dict(size=11))
    fig = go.Figure(go.Heatmap(
        x=ranges,
        y=velocities,
        z=power_db,
        colorscale=_COLORSCALE,
        zmin=-db_range,
        zmax=0,
        colorbar=cb,
        hoverongaps=False,
        hovertemplate='Range: %{x:.2f} m<br>Velocity: %{y:.2f} m/s<br>Power: %{z:.1f} dB<extra></extra>',
    ))
    fig.update_xaxes(title_text='Range (m)', **_AXIS_STYLE)
    fig.update_yaxes(title_text='Velocity (m/s)', **_AXIS_STYLE)
    fig.update_layout(
        title=dict(
            text=f'Range–Doppler — frame {frame_idx}',
            x=0.5,
            font=dict(size=15),
        ),
        **_LAYOUT_BASE,
    )
    fig.show()
