import numpy as np
import plotly.graph_objects as go

_LAYOUT_BASE = dict(
    template='plotly_dark',
    height=500,
    width=750,
    paper_bgcolor='#0d0d0d',
    plot_bgcolor='#0d0d0d',
)

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.08)',
    zeroline=True,
    zerolinecolor='rgba(255,255,255,0.25)',
    tickfont=dict(size=11),
)


def _to_db(x: np.ndarray, clip_db: float) -> np.ndarray:
    x_db = 10 * np.log10(x / x.max() + 1e-12)
    return np.clip(x_db, -clip_db, 0)


def plot_azimuth_spectrum(
    spectrum: np.ndarray,
    range_m: float,
    azimuth_fov_degrees: float = 60,
    db_range: float = 30,
) -> None:
    """
    2-D line plot of the MUSIC azimuth pseudo-spectrum for a single range bin.

    Parameters
    ----------
    spectrum : ndarray, shape [n_az]
        Raw MUSIC pseudo-spectrum slice at one range bin (not in dB).
    range_m : float
        Range in metres shown in the plot title.
    azimuth_fov_degrees : half-FOV used when scanning azimuth.
    db_range : dynamic range in dB (0 dB = peak).
    """
    n_az = spectrum.shape[0]
    az_deg = np.linspace(-azimuth_fov_degrees, azimuth_fov_degrees, n_az)
    power_db = _to_db(spectrum, db_range)

    fig = go.Figure(go.Scatter(
        x=az_deg,
        y=power_db,
        mode='lines',
        line=dict(color='#00ffff', width=1.5),
    ))

    fig.update_xaxes(title_text='Azimuth (degrees)', **_AXIS_STYLE)
    fig.update_yaxes(title_text='Power (dB)', range=[-db_range, 0], **_AXIS_STYLE)
    fig.update_layout(
        title=dict(
            text=f'MUSIC Azimuth Spectrum — Range {range_m:.2f} m',
            x=0.5,
            font=dict(size=15),
        ),
        **_LAYOUT_BASE,
    )
    fig.show()
