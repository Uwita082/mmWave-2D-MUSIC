import numpy as np
import plotly.graph_objects as go

_AXIS_STYLE = dict(
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
    plot_bgcolor='#0d0d0d',
    autosize=False,
)


def plot_cfar_detections(
    det: np.ndarray,
    range_resolution: float,
    azimuth_fov_degrees: float,
    az_step_deg: float = 0.25,
    frame_idx: int = 0,
    point_size: int = 6,
) -> None:
    """
    2-D Cartesian scatter of CFAR-detected points for a single frame.

    Parameters
    ----------
    det               : bool ndarray [frames, range_bins, az_bins]
    range_resolution  : metres per range bin
    azimuth_fov_degrees : half-FOV used to build scan_angles
    az_step_deg       : angular step used when building scan_angles
    frame_idx         : which frame to display (default 0)
    point_size        : marker size in pixels
    """
    frame_det = det[frame_idx]  # [range_bins, az_bins]
    range_indices, az_indices = np.where(frame_det)

    n_pts = len(range_indices)
    if n_pts == 0:
        print(f"Frame {frame_idx}: no CFAR detections.")
        return

    n_range_bins = det.shape[1]
    max_range = n_range_bins * range_resolution
    x_max = max_range * np.sin(np.deg2rad(azimuth_fov_degrees))

    scan_angles = np.arange(-azimuth_fov_degrees, azimuth_fov_degrees, az_step_deg)
    r = range_indices * range_resolution
    az_deg = scan_angles[az_indices]
    az_rad = np.deg2rad(az_deg)
    x = r * np.sin(az_rad)
    y = r * np.cos(az_rad)

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
        customdata=np.stack([r, az_deg], axis=1),
        marker=dict(
            size=point_size,
            color=r,
            cmin=0,
            cmax=max_range,
            colorscale='Jet',
            colorbar=dict(
                thickness=14,
                title=dict(text='Range (m)', side='right'),
                tickfont=dict(size=11),
            ),
            opacity=0.9,
        ),
        hovertemplate=(
            'x: %{x:.2f} m<br>'
            'y: %{y:.2f} m<br>'
            'range: %{customdata[0]:.2f} m<br>'
            'azimuth: %{customdata[1]:.2f}°'
            '<extra></extra>'
        ),
    )

    fig = go.Figure([radar_marker, detections])

    fig.update_layout(
        title=dict(
            text=f'CFAR Detections — frame {frame_idx}  ({n_pts} pts)',
            x=0.5,
            font=dict(size=15),
        ),
        xaxis=dict(title='X (m)', range=[-x_max, x_max], **_AXIS_STYLE),
        yaxis=dict(title='Y (m)', range=[0, max_range], scaleanchor='x', scaleratio=1, **_AXIS_STYLE),
        **_LAYOUT_BASE,
    )
    fig.show()
