import math
from typing import List, Optional

import numpy as np
from scipy.signal import find_peaks

from parser.parser_dca1000 import parse_mmwave_studio_adc
from plots.range import plot_range_fft
from utils.config import RAW_ADC_PATH, N_FRAMES, RX_CHANNELS, ADC_SAMPLES, \
    N_CHIRPS_LOOPS, RANGE_RESOLUTION, CALIB_ADC_PATH, CHIRP_CYCLE_TIME, TX_CHANNELS, WAVELENGTH
from plots.azimuth_spectrum import plot_azimuth_spectrum
from plots.range_doppler import plot_range_doppler
from plots.point_cloud import plot_point_cloud_rd
from plots.music_spectrum import plot_music_range_azimuth_heatmap
from scipy.linalg import svd
from cfar.cfar_2d import cfar_on_cube, peak_group, os_cfar_on_cube, ca_cfar_2d


class PointCloud1DMUSIC:
    azimuth_fov_degrees = 60
    """
    TX0 (First Checkbox) -> Connects to physical TX1 -> Azimuth
    TX1 (Second Checkbox) -> Connects to physical TX3 -> Azimuth
    """
    pos = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0]
    ])
    range_max = 4

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
                                        guard_doppler=1, guard_range=2,
                                        train_doppler=2, train_range=4,
                                        k_rank=None, pfa=1e-7
                                        )
        rd_det[:, :, :8] = False
        rd_det[:, :, -8:] = False

        # peak_group
        # for f in range(rd_det.shape[0]):
        #     rd_det[f] = peak_group(rd_power[f], rd_det[f], neighborhood=3)

        # Mask residual zero-Doppler line
        # Don't mask zero-Doppler - that's where the stationary human lives
        zero_dop = rd_det.shape[1] // 2
        # rd_det[:, zero_dop:zero_dop+1, :] = False
        # rd_det[:, zero_dop-1:zero_dop + 2, :] = False

        heatmap, _ = self.compute_music_heatmap(frame, frame_idx=0)
        plot_music_range_azimuth_heatmap(heatmap, self.azimuth_fov_degrees, frame_idx=0)

        clouds = self.point_cloud_from_rd_azimuth(frame, rd_det)
        plot_point_cloud_rd(clouds, frame_idx=0)

        rd_det, _ = self.cfar_detection(rd_power,
                                        guard_doppler=1, guard_range=8,
                                        train_doppler=2, train_range=32,
                                        k_rank=None, pfa=1e-24
                                        )
        rd_det[:, :, :8] = False
        rd_det[:, :, -8:] = False

        clouds = self.point_cloud_from_rd_azimuth(frame, rd_det)
        plot_point_cloud_rd(clouds, frame_idx=0)

    @staticmethod
    def static_clutter_removal(data: np.ndarray):
        """
        Input: data of shape [frames, chirps, rx_virtual, range]
        Output: data of shape [frames, chirps, rx_virtual, range]
        """
        mean_frames = data.mean(axis=1, keepdims=True)

        return data - mean_frames

    @staticmethod
    def range_fft(data: np.ndarray, use_window: bool):
        """
        Input: data of shape [frames, chirps, rx_virtual, adc_samples]
        Output: data of shape [frames, chirps, rx_virtual, range]
        """
        if use_window:
            hanning_window = np.hanning(data.shape[-1])
            data = data * hanning_window

        return np.fft.fft(data, axis=-1)

    @staticmethod
    def doppler_fft(data: np.ndarray, use_window: bool):
        """
        Input: data of shape [frames, chirps, rx_virtual, range]
        Output: data of shape [frames, doppler, rx_virtual, range]
        """
        if use_window:
            data = data.transpose([0, 2, 3, 1]) # [frames, rx_virtual, range, doppler]

            hanning_window = np.hanning(data.shape[-1])
            data = data * hanning_window

            data = data.transpose([0, 3, 1, 2]) # [frames, doppler, rx_virtual, range]

        doppler_fft = np.fft.fft(data, axis=1)
        return np.fft.fftshift(doppler_fft, axes=1)

    @staticmethod
    def phase_correction(frame: np.ndarray):
        """
        Input: data of shape [frames, doppler, rx_virtual, range]
        Output: data of shape [frames, doppler_corrected, rx_virtual, range]
        """

        T_loop = TX_CHANNELS * CHIRP_CYCLE_TIME

        # signed/centered doppler index: 0,1,...,N/2-1, -N/2,...,-1
        doppler_idx = np.arange(N_CHIRPS_LOOPS) - N_CHIRPS_LOOPS // 2
        f_doppler = doppler_idx / (N_CHIRPS_LOOPS * T_loop)

        correction = np.exp(-1j * 2 * np.pi * f_doppler * CHIRP_CYCLE_TIME)

        out = frame.copy()
        out[:, :, [4, 5, 6, 7], :] *= correction[None, :, None, None]

        return out

    @staticmethod
    def fb_spatial_smoothing(R, L):
        M = R.shape[0]
        K = M - L + 1
        Rf = np.zeros((L, L), dtype=complex)
        for i in range(K):
            Rf += R[i:i + L, i:i + L]
        Rf /= K
        J = np.fliplr(np.eye(L))  # exchange matrix
        Rb = J @ Rf.conj() @ J
        return 0.5 * (Rf + Rb)

    @staticmethod
    def music_algorithm(R, M, num_sources, scan_angles):
        """
        MUSIC angle estimation
        :param R: Covariance matrix
        :param M: Number of array elements
        :param num_sources: Estimated number of sources
        :param scan_angles: Range of search angles
        :return: Estimated angles
        """
        U, Lambda, Vh = svd(R)
        noise_subspace = Vh[num_sources:].conj().T  # Noise subspace
        angles_rad = np.radians(scan_angles)
        steering_matrix = np.exp(1j * np.pi * np.outer(np.arange(M), np.sin(angles_rad)))
        P_music = 1 / (np.linalg.norm(steering_matrix.conj().T @ noise_subspace, axis=1) ** 2)
        return P_music

    def visualization_azimuth(self, data: np.ndarray, detections_range_index: np.ndarray):
        # Input shape data: [range, n_az]
        # Input shape detections_range_index: boolean array over range bins
        for i in range(len(detections_range_index)):
            if detections_range_index[i]:
                plot_azimuth_spectrum(
                    data[i],
                    range_m=i * RANGE_RESOLUTION,
                    azimuth_fov_degrees=self.azimuth_fov_degrees,
                )

    @staticmethod
    def cfar_detection(data: np.ndarray, guard_doppler, guard_range,
                        train_doppler, train_range, k_rank, pfa):
        # Input shape: [frames, range, n_az]
        return os_cfar_on_cube(data, guard_doppler, guard_range, train_doppler, train_range, k_rank, pfa)

    def point_cloud(self, data: List[np.ndarray]):
        def to_xy(detections: np.ndarray):
            # detections shape: (N, 2) -> columns: range_idx, az_idx
            range_idx = detections[:, 0].astype(int)
            az_idx = detections[:, 1].astype(int)

            az_candidates = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)

            r = range_idx * RANGE_RESOLUTION
            az = np.deg2rad(az_candidates[az_idx])

            x = r * np.sin(az)
            y = r * np.cos(az)

            return np.stack([x, y], axis=1)

        frame_xy = []
        for frame_detections in data:
            xy = to_xy(frame_detections)
            frame_xy.append(xy)
        return frame_xy

    def steering_vector(self, azimuth):
        # azimuth: [n_az] in degrees
        az = np.deg2rad(np.atleast_1d(azimuth))

        # All elements are in the azimuth row (pos[:, 1] == 0), so only kx matters
        kx = np.pi * np.sin(az)  # [n_az]

        # phase: [rx, n_az]
        phase = self.pos[:, 0][:, None] * kx[None, :]

        return np.exp(1j * phase)  # [rx, n_az]

    def music_per_cell(self, snapshot: np.ndarray, scan_angles, L: int = 7, K: int = 2):
        """
        MUSIC on a single virtual-array snapshot (one [range, doppler] cell).
        Input : snapshot of shape [rx_virtual]  (complex, length 8)
        Output: (P_music [n_angles], scan_angles [n_angles])
        """
        R = np.outer(snapshot, snapshot.conj())  # rank-1, 8x8
        R_smooth = self.fb_spatial_smoothing(R, L)  # 4x4 after FB smoothing
        P_music = self.music_algorithm(R_smooth, L, K, scan_angles)
        return P_music, scan_angles

    def compute_music_heatmap(self, rd_cube: np.ndarray, frame_idx: int = 0):
        """
        Compute MUSIC pseudo-spectrum for every range bin using the peak-Doppler snapshot.

        Input : rd_cube [frames, doppler, rx_virtual, range]
        Output: (heatmap [n_range, n_az], scan_angles [n_az])
        """
        frame = rd_cube[frame_idx]          # [doppler, rx_virtual, range]
        n_doppler, n_rx, n_range = frame.shape

        doppler_power = (np.abs(frame) ** 2).sum(axis=1)   # [doppler, range]
        peak_dop = np.argmax(doppler_power, axis=0)         # [range]

        scan_angles = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)
        heatmap = np.zeros((n_range, len(scan_angles)))

        for r in range(n_range):
            x = frame[peak_dop[r], :, r]   # [rx_virtual]
            P, _ = self.music_per_cell(x, scan_angles)
            heatmap[r] = P

        return heatmap, scan_angles

    def point_cloud_from_rd_azimuth(self, rd_cube: np.ndarray,
                            rd_det: np.ndarray, K: int = 2):
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
        T_loop = TX_CHANNELS * CHIRP_CYCLE_TIME

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
    adc_data = parse_mmwave_studio_adc(RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, 2, N_FRAMES)

    p = PointCloud1DMUSIC()

    p.run(adc_data, True, True)

"""
Notes:
If you try to integrate over many frames to get a better covariance estimate, motion smears the steering vector and MUSIC degrades quickly. If you naively run MUSIC on a range–angle map without proper Doppler processing, fast scatterers spread their energy across angles.
"""
