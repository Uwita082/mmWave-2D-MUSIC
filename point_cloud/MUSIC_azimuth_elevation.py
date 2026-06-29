import math
from typing import List

import numpy as np
from scipy.linalg import svd
from scipy.signal import find_peaks

from parser.parser_dca1000 import parse_mmwave_studio_adc
from plots.range import plot_range_fft
from plots.range_doppler import plot_range_doppler
from plots.music_spectrum import plot_music_range_azimuth_heatmap
from plots.point_cloud import plot_point_cloud, plot_point_cloud_rd
from utils.config import (RAW_ADC_PATH, N_FRAMES, RX_CHANNELS,
                          ADC_SAMPLES, N_CHIRPS_LOOPS, RANGE_RESOLUTION,
                          CHIRP_CYCLE_TIME, WAVELENGTH)
from cfar.cfar_2d import os_cfar_on_cube, peak_group


class PointCloud2DMUSIC:
    azimuth_fov_degrees = 60
    elevation_fov_degrees = 30
    range_max = 4
    N_TX = 3

    """
    TX0 (First Checkbox)  -> Physical TX1 -> Azimuth
    TX1 (Second Checkbox) -> Physical TX3 -> Azimuth
    TX2 (Third Checkbox)  -> Physical TX2 -> Elevation

    Full virtual array (12 elements):
      Azimuth row  (y=0): indices 0-7  at x=0,1,2,3,4,5,6,7
      Elevation row (y=1): indices 8-11 at x=2,3,4,5

    Azimuth: high-resolution 1D MUSIC on the 8-element azimuth row.
    Elevation: monopulse from the 2-row phase difference at the known azimuth.
    """
    pos = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0],
        [2, 1], [3, 1], [4, 1], [5, 1],
    ])

    def run(self, adc_data: np.ndarray, static_removal: bool = False, phase_shift: bool = False):
        """Input: data of shape [frames, chirps, rx_virtual, adc_samples]"""
        frame = adc_data

        frame = self.range_fft(frame, True)

        max_bin = math.ceil(self.range_max / RANGE_RESOLUTION)
        frame = frame[:, :, :, :max_bin]

        plot_range_fft(frame[0, 0, 0, :], max_bin, RANGE_RESOLUTION)

        if static_removal:
            frame = self.static_clutter_removal(frame)
            plot_range_fft(frame[0, 0, 0, :], max_bin, RANGE_RESOLUTION)

        frame = self.doppler_fft(frame, True)

        if phase_shift:
            frame = self.phase_correction(frame)

        rd_power = (np.abs(frame) ** 2).sum(axis=2)  # [frames, doppler, range]

        plot_range_doppler(rd_power, frame_idx=0)

        rd_det, _ = self.cfar_detection(rd_power,
                                        guard_doppler=1, guard_range=1,
                                        train_doppler=2, train_range=2,
                                        k_rank=None, pfa=1e-2)

        for f in range(rd_det.shape[0]):
            rd_det[f] = peak_group(rd_power[f], rd_det[f], neighborhood=3)

        rd_det[:, :, :8] = False
        rd_det[:, :, -8:] = False

        zero_dop = rd_det.shape[1] // 2
        # rd_det[:, zero_dop - 1:zero_dop + 2, :] = False

        clouds = self.point_cloud_from_rd_azimuth(frame, rd_det)
        plot_point_cloud_rd(clouds, frame_idx=0)

        heatmap, _ = self.compute_music_heatmap(frame, frame_idx=0)
        plot_music_range_azimuth_heatmap(heatmap, self.azimuth_fov_degrees, frame_idx=0)

        clouds_rd = self.point_cloud_from_rd(frame, rd_det)
        clouds_xyz = self.point_cloud(clouds_rd)
        plot_point_cloud(clouds_xyz, frame_idx=0)

    @staticmethod
    def static_clutter_removal(data: np.ndarray) -> np.ndarray:
        """Input/Output: [frames, chirps, rx_virtual, range]"""
        mean_frames = data.mean(axis=1, keepdims=True)
        return data - mean_frames

    @staticmethod
    def range_fft(data: np.ndarray, use_window: bool) -> np.ndarray:
        """Input: [frames, chirps, rx_virtual, adc_samples] -> Output: [frames, chirps, rx_virtual, range]"""
        if use_window:
            hanning_window = np.hanning(data.shape[-1])
            data = data * hanning_window
        return np.fft.fft(data, axis=-1)

    @staticmethod
    def doppler_fft(data: np.ndarray, use_window: bool) -> np.ndarray:
        """Input: [frames, chirps, rx_virtual, range] -> Output: [frames, doppler, rx_virtual, range]"""
        if use_window:
            data = data.transpose([0, 2, 3, 1])  # [frames, rx_virtual, range, chirps]
            hanning_window = np.hanning(data.shape[-1])
            data = data * hanning_window
            data = data.transpose([0, 3, 1, 2])  # [frames, chirps, rx_virtual, range]

        doppler_fft = np.fft.fft(data, axis=1)
        return np.fft.fftshift(doppler_fft, axes=1)

    def phase_correction(self, frame: np.ndarray) -> np.ndarray:
        """
        Input/Output: [frames, doppler, rx_virtual, range]
        TX1 (virtual indices 4-7)  gets 1x per-chirp phase correction.
        TX2 (virtual indices 8-11) gets 2x per-chirp phase correction.
        """
        T_loop = self.N_TX * CHIRP_CYCLE_TIME
        doppler_idx = np.arange(N_CHIRPS_LOOPS) - N_CHIRPS_LOOPS // 2
        f_doppler = doppler_idx / (N_CHIRPS_LOOPS * T_loop)

        corr1 = np.exp(-1j * 2 * np.pi * f_doppler * CHIRP_CYCLE_TIME)
        corr2 = corr1 ** 2

        out = frame.copy()
        out[:, :, [4, 5, 6, 7], :] *= corr1[None, :, None, None]
        out[:, :, [8, 9, 10, 11], :] *= corr2[None, :, None, None]
        return out

    @staticmethod
    def fb_spatial_smoothing(R: np.ndarray, L: int) -> np.ndarray:
        M = R.shape[0]
        K = M - L + 1
        Rf = np.zeros((L, L), dtype=complex)
        for i in range(K):
            Rf += R[i:i + L, i:i + L]
        Rf /= K
        J = np.fliplr(np.eye(L))
        Rb = J @ Rf.conj() @ J
        return 0.5 * (Rf + Rb)

    @staticmethod
    def music_algorithm(R: np.ndarray, M: int, num_sources: int,
                        scan_angles: np.ndarray) -> np.ndarray:
        U, Lambda, Vh = svd(R)
        noise_subspace = Vh[num_sources:].conj().T
        angles_rad = np.radians(scan_angles)
        steering_matrix = np.exp(1j * np.pi * np.outer(np.arange(M), np.sin(angles_rad)))
        P_music = 1 / (np.linalg.norm(steering_matrix.conj().T @ noise_subspace, axis=1) ** 2)
        return P_music

    def music_per_cell(self, snapshot: np.ndarray, scan_angles: np.ndarray,
                       L: int = 7, K: int = 4):
        """1D MUSIC on the 8-element azimuth row (virtual indices 0-7)."""
        x = snapshot[:8]
        R = np.outer(x, x.conj())
        R_smooth = self.fb_spatial_smoothing(R, L)
        P_music = self.music_algorithm(R_smooth, L, K, scan_angles)
        return P_music, scan_angles

    @staticmethod
    def estimate_elevation(snapshot: np.ndarray, az_deg: float) -> float:
        """Monopulse elevation from the 2-row URA."""
        az = np.deg2rad(az_deg)
        kx = np.pi * np.sin(az)
        a_az = np.exp(-1j * np.arange(4) * kx)  # 4 URA columns
        row0 = np.sum(snapshot[[2, 3, 4, 5]] * a_az)
        row1 = np.sum(snapshot[[8, 9, 10, 11]] * a_az)
        phase_diff = np.angle(row0 * np.conj(row1))
        sin_el = phase_diff / np.pi
        return np.rad2deg(np.arcsin(np.clip(sin_el, -1, 1)))

    def compute_music_heatmap(self, rd_cube: np.ndarray, frame_idx: int = 0):
        """
        Range-Azimuth MUSIC heatmap using the peak-Doppler snapshot.

        Input : rd_cube [frames, doppler, rx_virtual, range]
        Output: (heatmap [n_range, n_az], scan_angles [n_az])
        """
        frame = rd_cube[frame_idx]          # [doppler, rx_virtual, range]
        n_doppler, n_rx, n_range = frame.shape

        doppler_power = (np.abs(frame) ** 2).sum(axis=1)
        peak_dop = np.argmax(doppler_power, axis=0)

        scan_angles = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)
        heatmap = np.zeros((n_range, len(scan_angles)))

        for r in range(n_range):
            x = frame[peak_dop[r], :, r]
            P, _ = self.music_per_cell(x, scan_angles)
            heatmap[r] = P

        return heatmap, scan_angles

    @staticmethod
    def cfar_detection(data: np.ndarray, guard_doppler, guard_range,
                       train_doppler, train_range, k_rank, pfa):
        """Input: [frames, doppler, range]"""
        return os_cfar_on_cube(data, guard_doppler, guard_range,
                               train_doppler, train_range, k_rank, pfa)

    def point_cloud_from_rd(self, rd_cube: np.ndarray,
                            rd_det: np.ndarray, K: int = 4) -> List[np.ndarray]:
        """
        1D azimuth MUSIC + monopulse elevation per CFAR detection.

        Inputs
        ------
        rd_cube : [frames, doppler, rx_virtual, range]  (phase-corrected)
        rd_det  : [frames, doppler, range]               bool, from CFAR

        Returns
        -------
        list of length n_frames; each entry is (n_pts, 5) with columns
        (range_m, velocity_m_s, azimuth_deg, elevation_deg, power_db).
        """
        n_frames, n_doppler, _, _ = rd_cube.shape
        T_loop = self.N_TX * CHIRP_CYCLE_TIME

        doppler_idx = np.arange(n_doppler) - n_doppler // 2
        v_axis = doppler_idx * WAVELENGTH / (2 * n_doppler * T_loop)

        scan_angles = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)

        out = []
        for f in range(n_frames):
            det = rd_det[f]
            d_idx, r_idx = np.where(det)
            pts = []
            for d, r in zip(d_idx, r_idx):
                x = rd_cube[f, d, :, r]           # [12]
                P, scan = self.music_per_cell(x, scan_angles)
                P_db = 10 * np.log10(P + 1e-12)
                peak_idx, _ = find_peaks(P_db, prominence=6)
                order = np.argsort(P[peak_idx])[::-1][:K]
                power_db = 10 * np.log10(np.sum(np.abs(x[:8]) ** 2) + 1e-12)
                for pk in peak_idx[order]:
                    az = scan[pk]
                    el = self.estimate_elevation(x, az)
                    if abs(el) > self.elevation_fov_degrees:
                        continue
                    pts.append((r * RANGE_RESOLUTION, v_axis[d], az, el, power_db))
            out.append(np.array(pts) if pts else np.empty((0, 5)))
        return out

    def point_cloud(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """Convert (range_m, velocity_m_s, az_deg, el_deg, power_db) to
        (x, y, z, range_m, velocity_m_s, az_deg, el_deg, power_db)."""
        def to_xyz_full(detections: np.ndarray) -> np.ndarray:
            r_m      = detections[:, 0]
            vel      = detections[:, 1]
            az_deg   = detections[:, 2]
            el_deg   = detections[:, 3]
            power_db = detections[:, 4]

            az_rad = np.deg2rad(az_deg)
            el_rad = np.deg2rad(el_deg)

            x = r_m * np.cos(el_rad) * np.sin(az_rad)
            y = r_m * np.cos(el_rad) * np.cos(az_rad)
            z = r_m * np.sin(el_rad)

            return np.stack([x, y, z, r_m, vel, az_deg, el_deg, power_db], axis=1)

        out = []
        for frame_detections in data:
            if frame_detections.size > 0:
                out.append(to_xyz_full(frame_detections))
            else:
                out.append(np.empty((0, 8)))
        return out

    def point_cloud_from_rd_azimuth(self, rd_cube: np.ndarray,
                            rd_det: np.ndarray, K: int = 3):
        """
        Run MUSIC on each surviving (range, doppler) detection.

        Inputs
        ------
        rd_cube  : [frames, doppler, rx_virtual, range]   (phase-corrected)
        rd_det   : [frames, doppler, range]               bool, from CFAR

        Returns
        -------
        list of length n_frames; each entry is (n_pts, 3) with columns
        (range_m, velocity_m_s, azimuth_deg).
        """

        n_frames, n_doppler, _, _ = rd_cube.shape
        T_loop = 2 * CHIRP_CYCLE_TIME

        # Velocity axis, matching the fftshift'd Doppler bins
        doppler_idx = np.arange(n_doppler) - n_doppler // 2
        v_axis = doppler_idx * WAVELENGTH / (2 * n_doppler * T_loop)

        scan_angles = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)

        out = []
        for f in range(n_frames):
            det = rd_det[f]

            d_idx, r_idx = np.where(det)
            pts = []
            for d, r in zip(d_idx, r_idx):
                x = rd_cube[f, d, :, r]
                P, scan = self.music_per_cell(x, scan_angles)
                P_db = 10 * np.log10(P + 1e-12)
                peak_idx, _ = find_peaks(P_db, prominence=6)
                # take up to K strongest
                order = np.argsort(P[peak_idx])[::-1][:K]
                power_db = 10 * np.log10(np.sum(np.abs(x) ** 2) + 1e-12)
                for pk in peak_idx[order]:
                    pts.append((r * RANGE_RESOLUTION, v_axis[d], scan[pk], power_db))
            out.append(np.array(pts) if pts else np.empty((0, 4)))
        return out


if __name__ == '__main__':
    adc_data = parse_mmwave_studio_adc(RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, 3, N_FRAMES)

    p = PointCloud2DMUSIC()
    p.run(adc_data, False, True)
