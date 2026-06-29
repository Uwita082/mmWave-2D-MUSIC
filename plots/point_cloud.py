from typing import List, Optional

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

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.08)',
    zeroline=True,
    zerolinecolor='rgba(255,255,255,0.25)',
    tickfont=dict(size=11),
    backgroundcolor='#0d0d0d',
)

_AXIS_STYLE_2D = dict(
    showgrid=True,
    gridcolor='rgba(255,255,255,0.08)',
    zeroline=True,
    zerolinecolor='rgba(255,255,255,0.25)',
    tickfont=dict(size=11),
)

_LAYOUT_BASE = dict(
    template='plotly_dark',
    height=700,
    width=820,
    paper_bgcolor='#0d0d0d',
)

_MIN_RANGE = 4.0  # minimum axis extent in metres
_SENSOR_HEIGHT = 1.4  # sensor height above ground [m]


def _origin_circle(z_center: float, radius: float = 0.25) -> go.Scatter3d:
    """Yellow ring marking the sensor position."""
    theta = np.linspace(0, 2 * np.pi, 64)
    return go.Scatter3d(
        x=radius * np.cos(theta),
        y=np.zeros(64),
        z=np.full(64, z_center),
        mode='lines',
        line=dict(color='yellow', width=4),
        showlegend=False,
        hoverinfo='skip',
    )


def plot_point_cloud(
    point_clouds: List[np.ndarray],
    frame_idx: int = 0,
    point_size: int = 4,
    sensor_height: float = _SENSOR_HEIGHT,
) -> None:
    """
    3-D scatter of one frame's point cloud.

    Parameters
    ----------
    point_clouds  : list of ndarray, each shape (N, 8)
                    columns = x, y, z [m], range_m, velocity_m_s,
                              azimuth_deg, elevation_deg, power_db.
                    Falls back to (N, 3) xyz-only with range colouring.
    frame_idx     : which frame to display
    point_size    : base marker size in pixels (scaled by power when available)
    sensor_height : height of sensor above ground [m]
    """
    pts = point_clouds[frame_idx]
    if pts.size == 0:
        print(f"Frame {frame_idx}: no detections.")
        return

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2] + sensor_height
    n = len(x)

    has_meta = pts.shape[1] >= 8

    if has_meta:
        r_m      = pts[:, 3]
        vel      = pts[:, 4]
        az_deg   = pts[:, 5]
        el_deg   = pts[:, 6]
        power_db = pts[:, 7]

        v_abs = max(abs(float(vel.min())), abs(float(vel.max())), 0.01)
        p_min, p_max = float(power_db.min()), float(power_db.max())
        sizes = (4 + 10 * (power_db - p_min) / (p_max - p_min)
                 if p_max > p_min else np.full(n, point_size, dtype=float))

        color          = vel
        cmin, cmax     = -v_abs, v_abs
        colorbar_title = 'Velocity (m/s)'
        customdata     = np.stack([r_m, vel, az_deg, el_deg, power_db], axis=1)
        hovertemplate  = (
            'x: %{x:.3f} m<br>'
            'y: %{y:.3f} m<br>'
            'height: %{z:.3f} m<br>'
            'range: %{customdata[0]:.2f} m<br>'
            'velocity: %{customdata[1]:.2f} m/s<br>'
            'azimuth: %{customdata[2]:.1f}°<br>'
            'elevation: %{customdata[3]:.1f}°<br>'
            'power: %{customdata[4]:.1f} dB'
            '<extra></extra>'
        )
    else:
        r_m            = np.sqrt(x ** 2 + y ** 2 + (z - sensor_height) ** 2)
        sizes          = point_size
        color          = r_m
        cmin, cmax     = float(r_m.min()), float(r_m.max())
        colorbar_title = 'Range (m)'
        customdata     = None
        hovertemplate  = (
            'x: %{x:.3f} m<br>y: %{y:.3f} m<br>'
            'height: %{z:.3f} m<extra></extra>'
        )

    x_range = [min(float(x.min()), -_MIN_RANGE), max(float(x.max()), _MIN_RANGE)]
    y_range = [0.0, max(float(y.max()), _MIN_RANGE)]
    z_range = [min(float(z.min()), 0.0), max(float(z.max()), sensor_height + _MIN_RANGE)]

    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=sizes,
            color=color,
            colorscale=_COLORSCALE,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                thickness=14,
                title=dict(text=colorbar_title, side='right'),
                tickfont=dict(size=11),
            ),
            opacity=0.85,
        ),
        customdata=customdata,
        hovertemplate=hovertemplate,
    )

    fig = go.Figure([scatter, _origin_circle(z_center=sensor_height)])
    fig.update_layout(
        title=dict(text=f'Point Cloud — frame {frame_idx}  ({n} pts)',
                   x=0.5, font=dict(size=15)),
        scene=dict(
            xaxis=dict(title='X (m)', range=x_range, **_AXIS_STYLE),
            yaxis=dict(title='Y (m)', range=y_range, **_AXIS_STYLE),
            zaxis=dict(title='Height (m)', range=z_range, **_AXIS_STYLE),
            bgcolor='#0d0d0d',
            aspectmode='cube',
        ),
        **_LAYOUT_BASE,
    )
    fig.show()


def plot_point_cloud_rd(
    clouds: List[np.ndarray],
    frame_idx: int = 0,
    point_size: int = 6,
    velocity_range: Optional[tuple] = None,
) -> None:
    """
    2-D top-down scatter of one frame's point cloud from Range-Doppler MUSIC.

    Parameters
    ----------
    clouds       : list of ndarray, each shape (N, 4)
                   columns = (range_m, velocity_m_s, azimuth_deg, power_db)
    frame_idx    : which frame to display
    point_size   : marker size in pixels
    velocity_range : (v_min, v_max) for colorbar; auto-scaled if None
    """
    pts = clouds[frame_idx]  # (N, 4)
    if pts.size == 0:
        print(f"Frame {frame_idx}: no detections.")
        return

    r_m = pts[:, 0]
    v_ms = pts[:, 1]
    az_rad = np.deg2rad(pts[:, 2])
    power_db = pts[:, 3]

    x = r_m * np.sin(az_rad)
    y = r_m * np.cos(az_rad)

    v_min, v_max = (v_ms.min(), v_ms.max()) if velocity_range is None else velocity_range
    max_range = float(r_m.max()) if r_m.size else _MIN_RANGE
    x_abs = max(float(np.abs(x).max()), _MIN_RANGE)

    # Scale marker size by power: map [p_min, p_max] → [6, 18] px
    p_min, p_max = float(power_db.min()), float(power_db.max())
    if p_max > p_min:
        sizes = 6 + 12 * (power_db - p_min) / (p_max - p_min)
    else:
        sizes = np.full(len(power_db), point_size, dtype=float)

    radar_marker = go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(symbol='triangle-up', size=14, color='yellow'),
        name='Radar',
        hovertemplate='Radar (0, 0)<extra></extra>',
    )

    detections = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Detections',
        customdata=np.stack([r_m, v_ms, pts[:, 2], power_db], axis=1),
        marker=dict(
            size=sizes,
            color=v_ms,
            colorscale=_COLORSCALE,
            cmin=-max(abs(v_min), abs(v_max)),
            cmax=max(abs(v_min), abs(v_max)),
            colorbar=dict(
                thickness=14,
                title=dict(text='Velocity (m/s)', side='right'),
                tickfont=dict(size=11),
            ),
            opacity=0.9,
        ),
        hovertemplate=(
            'x: %{x:.2f} m<br>'
            'y: %{y:.2f} m<br>'
            'range: %{customdata[0]:.2f} m<br>'
            'velocity: %{customdata[1]:.2f} m/s<br>'
            'azimuth: %{customdata[2]:.1f}°<br>'
            'power: %{customdata[3]:.1f} dB'
            '<extra></extra>'
        ),
    )

    fig = go.Figure([radar_marker, detections])
    fig.update_layout(
        title=dict(
            text=f'Point Cloud — frame {frame_idx}  ({len(x)} pts)',
            x=0.5,
            font=dict(size=15),
        ),
        xaxis=dict(title='X (m)', range=[-x_abs, x_abs], **_AXIS_STYLE_2D),
        yaxis=dict(title='Y (m)', range=[0, max(max_range, _MIN_RANGE)],
                   scaleanchor='x', scaleratio=1, **_AXIS_STYLE_2D),
        **_LAYOUT_BASE,
    )
    fig.show()
