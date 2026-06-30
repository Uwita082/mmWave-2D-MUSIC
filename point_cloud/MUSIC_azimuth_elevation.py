import math
from typing import List, Optional

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
    """
    End-to-end mmWave radar signal processing pipeline that produces a full 3D point cloud
    (x, y, z) from raw ADC samples captured with a 3-TX TDM-MIMO radar.

    Azimuth angle is resolved using 1-D MUSIC on the 8-element azimuth virtual ULA row.
    Elevation angle is resolved using monopulse processing on the 2-row virtual aperture,
    exploiting the inter-row phase difference at the azimuth angle already estimated by MUSIC.

    The processing chain is as follows:
        Range FFT  ->  Static Clutter Removal  ->  Doppler FFT
        ->  TDM-MIMO Phase Correction  ->  2-D CFAR Detection  ->  Peak Grouping
        ->  MUSIC Azimuth Estimation  ->  Monopulse Elevation Estimation
        ->  3D Point Cloud Assembly (x, y, z)

    Full virtual array (12 elements, 3-TX TDM-MIMO):
        Azimuth row   (y=0): virtual elements 0-7   at x = 0, 1, 2, 3, 4, 5, 6, 7
        Elevation row (y=1): virtual elements 8-11  at x = 2, 3, 4, 5

    Transmitter-to-port mapping (as configured in mmWave Studio):
        TX0 (First Checkbox)  -> Physical TX1 -> Azimuth   (virtual elements 0-3)
        TX1 (Second Checkbox) -> Physical TX3 -> Azimuth   (virtual elements 4-7)
        TX2 (Third Checkbox)  -> Physical TX2 -> Elevation (virtual elements 8-11)

    IMPORTANT: In mmWave Studio the chirp firing order must be TX0 -> TX2 -> TX1
    (first chirp fires TX0, second fires TX2, third fires TX1). This ordering ensures
    that the first 8 virtual elements form the contiguous azimuth ULA and the last 4
    virtual elements form the elevation row at y=1.
    """

    # Maximum azimuth half-angle in degrees; the MUSIC pseudo-spectrum is scanned
    # symmetrically from -azimuth_fov_degrees to +azimuth_fov_degrees.
    azimuth_fov_degrees = 60

    # Maximum elevation half-angle in degrees; detections whose estimated elevation
    # magnitude exceeds this value are rejected as physically implausible.
    elevation_fov_degrees = 30

    # Number of active transmit channels; determines the inter-TX loop time
    # T_loop = N_TX * T_chirp used in the phase correction and velocity axis scaling.
    N_TX = 3

    # Full 12-element virtual array positions in half-wavelength units.
    # Column 0: x-coordinate along the azimuth axis.
    # Column 1: y-coordinate along the elevation axis (0 for azimuth row, 1 for elevation row).
    # The azimuth row (elements 0-7) is a uniform linear array at y=0 with unit x-spacing,
    # satisfying the spatial Nyquist criterion and preventing aliasing up to +-90 deg in azimuth.
    # The elevation row (elements 8-11) lies at y=1 and spans x=2..5, overlapping the
    # central four columns of the azimuth row to enable matched monopulse processing.
    pos = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0],
        [2, 1], [3, 1], [4, 1], [5, 1],
    ])

    def run(self, adc_data: np.ndarray, static_removal: bool = False,
            phase_shift: bool = False, mask_zero_doppler: bool = False,
            num_targets: int = 4, peak_group_neighborhood: int = 3,
            range_max: Optional[float] = 4.0):
        """
        Executes the full 3D radar signal processing pipeline on a batch of raw ADC frames.

        The pipeline converts raw complex ADC samples into a 3D point cloud (x, y, z) by
        sequentially applying: Range FFT, optional static clutter removal, Doppler FFT,
        optional 3-TX TDM-MIMO phase correction, 2-D CFAR detection, peak grouping, MUSIC
        azimuth estimation, and monopulse elevation estimation. Intermediate results are
        visualised through dedicated plotting routines at key stages.

        Parameters
        ----------
        adc_data               : ndarray, shape (frames, chirps, rx_virtual, adc_samples)
            Raw complex ADC data as captured by the radar front-end.
        static_removal         : bool, default False
            If True, subtracts the intra-frame chirp mean to suppress static clutter.
            Disable when the targets of interest are stationary.
        phase_shift            : bool, default False
            If True, applies the 3-TX TDM-MIMO inter-TX phase correction after the
            Doppler FFT. Enable whenever targets are moving to prevent an angular bias
            introduced by the velocity-induced inter-chirp phase ramp.
        mask_zero_doppler      : bool, default False
            If True, suppresses the three central Doppler bins (zero-velocity line and
            its immediate neighbours) in the CFAR detection mask to remove residual
            static returns that survived clutter removal.
        num_targets            : int, default 4
            Estimated number of simultaneous signal sources present in any single
            range-Doppler cell. Passed to MUSIC as K; must be strictly less than the
            spatial smoothing sub-array length L (default L=7).
        peak_group_neighborhood: int, default 3
            Neighbourhood radius (in bins) for post-CFAR peak grouping. Each cluster of
            adjacent detections is reduced to the local power-maximum cell. Set to 0 to
            retain all individual CFAR detections without grouping.
        range_max              : float or None, default 4.0
            Maximum range of interest in metres. The range-FFT output is cropped to the
            corresponding bin index before further processing. Set to None to retain all
            range bins up to the maximum unambiguous range.
        """
        # Assign the input ADC data to the working buffer.
        frame = adc_data

        # --- Stage 1: Range FFT ---
        # Apply the Range FFT (fast-time FFT) along the ADC sample axis.
        # Each beat-frequency tone in the ADC record maps to a discrete range bin whose
        # index is proportional to the round-trip delay of the target.
        frame = self.range_fft(frame, True)

        # --- Stage 2: Range Gate Selection ---
        # Determine the maximum range bin index to retain.
        if range_max is not None and range_max < ADC_SAMPLES * RANGE_RESOLUTION:
            # Convert the physical range limit to the corresponding bin index.
            # math.ceil ensures that a target at exactly range_max is included in the window.
            max_bin = math.ceil(range_max / RANGE_RESOLUTION)
            frame = frame[:, :, :, :max_bin]
        else:
            # If no range limit is specified (or it exceeds the unambiguous range),
            # retain all available range bins.
            max_bin = ADC_SAMPLES

        # Plot the Range FFT magnitude spectrum before clutter removal to show the raw range
        # profile. A sharp peak in the magnitude spectrum indicates a reflector at that range.
        plot_range_fft(frame[0, 0, 0, :], max_bin, RANGE_RESOLUTION)

        # --- Stage 3: Static Clutter Removal ---
        # Static scatterers, such as walls, ceiling, floor, furniture, and the radar mounting
        # structure, produce nearly identical signals on every chirp within a frame.
        # Subtracting the intra-frame chirp mean removes their DC contribution, leaving only
        # the time-varying (moving-target) components.
        if static_removal:
            frame = self.static_clutter_removal(frame)

            # Re-plot the range profile after clutter removal to confirm that static peaks
            # have been suppressed and moving-target peaks are more prominent.
            plot_range_fft(frame[0, 0, 0, :], max_bin, RANGE_RESOLUTION)

        # --- Stage 4: Doppler FFT ---
        # Apply the Doppler FFT (slow-time FFT) across the chirp axis.
        # A moving target introduces a linearly increasing inter-chirp phase shift
        # proportional to its radial velocity; the Doppler FFT maps this progression
        # to a unique velocity bin. An fftshift centres zero Doppler on the axis.
        frame = self.doppler_fft(frame, True)

        # --- Stage 5: TDM-MIMO Phase Correction ---
        # In 3-TX TDM-MIMO, each transmitter fires in a dedicated slot within T_chirp.
        # A moving target accumulates additional inter-chirp phase proportional to its
        # radial velocity and the inter-TX delay: n * f_d * T_chirp for TX index n.
        # TX1 (n=1) requires one correction phasor; TX2 (n=2) requires two.
        # Without correction, the angular estimate is shifted by a velocity-dependent bias.
        if phase_shift:
            frame = self.phase_correction(frame)

        # --- Stage 6: Range-Doppler Power Map ---
        # Compute the incoherent power sum across all 12 virtual receivers.
        # Non-coherent integration across the receiver axis improves SNR and produces a
        # 2-D (Doppler x range) power map suitable for adaptive CFAR thresholding.
        rd_power = (np.abs(frame) ** 2).sum(axis=2)  # shape: (frames, doppler, range)

        # Plot the range-Doppler heatmap for visual inspection.
        # A positive Doppler bin indicates a target receding from the radar;
        # a negative bin indicates a target approaching.
        plot_range_doppler(rd_power, frame_idx=0)

        # --- Stage 7: 2-D CFAR Detection ---
        # Apply OS-CFAR to the range-Doppler power map to identify target-bearing cells
        # while maintaining an approximately constant probability of false alarm across
        # varying local noise and clutter levels.
        rd_det, _ = self.cfar_detection(rd_power,
                                        guard_doppler=1, guard_range=1,
                                        train_doppler=2, train_range=2,
                                        k_rank=None, pfa=1e-2)

        # --- Stage 8: Peak Grouping ---
        # Replace each cluster of adjacent detections with a single cell at the local
        # power maximum, reducing multiple hits from the same physical target to one
        # representative detection point. Applied per frame.
        if peak_group_neighborhood > 0:
            for f in range(rd_det.shape[0]):
                rd_det[f] = peak_group(rd_power[f], rd_det[f], neighborhood=peak_group_neighborhood)

        # --- Stage 9: Edge Range Bin Suppression ---
        # Blank the first and last 8 range bins in the detection mask.
        # These bins are dominated by electronic artefacts (DC offset, receiver-chain
        # transients) rather than genuine reflected target energy.
        rd_det[:, :, :8] = False   # suppress near-range edge bins
        rd_det[:, :, -8:] = False  # suppress far-range edge bins

        # --- Stage 10: Zero-Doppler Masking ---
        # Optionally suppress the three central Doppler bins to remove residual static
        # returns not fully cancelled by clutter removal. Enable only when all targets
        # of interest are known to be moving.
        if mask_zero_doppler:
            zero_dop = rd_det.shape[1] // 2  # index of the DC (zero-velocity) Doppler bin
            rd_det[:, zero_dop - 1:zero_dop + 2, :] = False

        # --- Stage 11: Azimuth-Only Point Cloud (Intermediate Visualisation) ---
        # Run MUSIC on the 8-element azimuth row for each CFAR detection to produce a
        # (range, velocity, azimuth, power) point cloud. This intermediate output allows
        # validation of the azimuth estimation stage before elevation is incorporated.
        clouds = self.point_cloud_from_rd_azimuth(frame, rd_det, K=num_targets)
        plot_point_cloud_rd(clouds, frame_idx=0)

        # --- Stage 12: MUSIC Range-Azimuth Heatmap ---
        # Compute and display the MUSIC pseudo-spectrum heatmap over the full range-azimuth
        # plane to validate angular localisation and observe the effect of spatial smoothing.
        heatmap, _ = self.compute_music_heatmap(frame, frame_idx=0, num_targets=num_targets)
        plot_music_range_azimuth_heatmap(heatmap, self.azimuth_fov_degrees, frame_idx=0)

        # --- Stage 13: Full 3D Point Cloud Assembly ---
        # Run MUSIC azimuth estimation and monopulse elevation estimation for each CFAR
        # detection, then convert the spherical coordinates (range, azimuth, elevation) to
        # Cartesian (x, y, z) for the final 3D point cloud output.
        clouds_rd = self.point_cloud_from_rd(frame, rd_det, K=num_targets)
        clouds_xyz = self.point_cloud(clouds_rd)
        plot_point_cloud(clouds_xyz, frame_idx=0)

    @staticmethod
    def static_clutter_removal(data: np.ndarray) -> np.ndarray:
        """
        Removes static clutter by subtracting the intra-frame mean across the slow-time
        (chirp) axis.

        Static scatterers, such as walls, ceiling, floor, furniture, and the radar mounting
        structure, produce nearly identical baseband signals on every chirp within a frame
        because their range and reflectivity do not change appreciably over the frame duration.
        Computing the mean along the chirp axis yields a clean estimate of the static
        background for each (receiver channel, range bin) cell. Subtracting it leaves only
        the time-varying component, i.e., the signal attributed to moving targets.

        This operation is equivalent to a single-tap high-pass filter along the slow-time
        axis and is the simplest form of Moving Target Indication (MTI).

        Input:  data of shape (frames, chirps, rx_virtual, range)
        Output: data of shape (frames, chirps, rx_virtual, range)
        """
        # Compute the chirp mean within each frame, independently per receiver channel and
        # range bin. keepdims=True retains the chirp axis dimension so that NumPy broadcasting
        # subtracts the mean correctly from every chirp snapshot without an explicit reshape.
        mean_frames = data.mean(axis=1, keepdims=True)

        # Subtract the static background estimate from every chirp snapshot.
        # The result is zero-mean along the chirp axis, suppressing DC (stationary) returns.
        return data - mean_frames

    @staticmethod
    def range_fft(data: np.ndarray, use_window: bool) -> np.ndarray:
        """
        Applies the Range FFT (fast-time FFT) to convert raw ADC samples into the range domain.

        After down-conversion and mixing, the beat signal's instantaneous frequency is
        proportional to the target's round-trip delay (i.e., its range). The FFT maps each
        beat frequency to a discrete range bin. A Hanning window, when enabled, reduces
        spectral leakage at the cost of approximately 1.5x reduction in range resolution
        relative to a rectangular window.

        Input:  data of shape (frames, chirps, rx_virtual, adc_samples)
        Output: data of shape (frames, chirps, rx_virtual, range)
        """
        if use_window:
            # Construct a Hanning window of length equal to the number of ADC samples.
            # Coefficients taper from zero at both endpoints to one at the centre.
            hanning_window = np.hanning(data.shape[-1])

            # Apply the window along the last axis (ADC samples / fast time) via broadcasting.
            # The 1-D window is broadcast across all frames, chirps, and receiver channels.
            data = data * hanning_window

        # Apply the DFT along the last axis (fast time / ADC samples).
        # Each output bin corresponds to one range gate.
        return np.fft.fft(data, axis=-1)

    @staticmethod
    def doppler_fft(data: np.ndarray, use_window: bool) -> np.ndarray:
        """
        Applies the Doppler FFT (slow-time FFT) to convert inter-chirp phase variations
        into the velocity (Doppler) domain.

        A moving target accumulates a linearly increasing phase from chirp to chirp,
        proportional to its radial velocity. The FFT across the chirp index maps each
        velocity to a unique Doppler bin. An fftshift recentres zero Doppler on the axis,
        placing approaching targets (negative velocity) on the left and receding targets
        (positive velocity) on the right.

        A Hanning window along the chirp axis optionally reduces Doppler sidelobes at the
        cost of a slight broadening of the velocity resolution cell.

        Input:  data of shape (frames, chirps, rx_virtual, range)
        Output: data of shape (frames, doppler, rx_virtual, range)
        """
        if use_window:
            # Transpose to move the chirp axis (axis 1) to the last position for windowing.
            # Resulting shape: (frames, rx_virtual, range, chirps)
            data = data.transpose([0, 2, 3, 1])

            # Construct a Hanning window of length equal to the number of chirps.
            hanning_window = np.hanning(data.shape[-1])

            # Apply the window along the chirp axis (now the last axis).
            data = data * hanning_window

            # Transpose back to restore the canonical dimension ordering.
            # Resulting shape: (frames, chirps, rx_virtual, range)
            data = data.transpose([0, 3, 1, 2])

        # Apply the DFT along axis 1 (slow time / chirps).
        # Each output bin corresponds to one Doppler (velocity) bin.
        doppler_fft = np.fft.fft(data, axis=1)

        # Shift zero frequency to the centre of the Doppler axis, converting the default
        # [0, ..., N-1] bin ordering to the symmetric [-N/2, ..., N/2-1] convention.
        return np.fft.fftshift(doppler_fft, axes=1)

    def phase_correction(self, frame: np.ndarray) -> np.ndarray:
        """
        Compensates for the inter-transmitter phase ramp introduced by target motion in
        3-TX TDM-MIMO radar.

        In 3-TX TDM-MIMO, each transmitter fires in a dedicated slot within the chirp cycle
        of duration T_chirp. For a moving target, the beat-signal phase changes between
        successive TX firings. The accumulated phase error for TX index n is:
            phi_n(k) = -2*pi * f_d[k] * n * T_chirp
        where f_d[k] is the Doppler frequency of the k-th bin. Without correction, this
        velocity-induced phase produces a fictitious angular offset in the virtual-array
        steering vector.

        Correction phasors per TX index:
            TX0 (n=0, virtual elements 0-3):  no correction needed (reference transmitter)
            TX1 (n=1, virtual elements 4-7):  corr1 = exp(-j * 2*pi * f_d * T_chirp)
            TX2 (n=2, virtual elements 8-11): corr2 = corr1^2 = exp(-j * 2*pi * f_d * 2*T_chirp)

        Input:  data of shape (frames, doppler, rx_virtual, range)
        Output: data of shape (frames, doppler_corrected, rx_virtual, range)
        """
        # Compute the inter-TX loop time: 3 TX channels each occupying one T_chirp slot.
        T_loop = self.N_TX * CHIRP_CYCLE_TIME

        # Construct the signed (centred) Doppler bin index array matching the fftshift convention.
        # Subtracting N_CHIRPS_LOOPS // 2 maps bin 0 to -N/2 and bin N-1 to N/2-1.
        doppler_idx = np.arange(N_CHIRPS_LOOPS) - N_CHIRPS_LOOPS // 2

        # Convert Doppler bin indices to physical Doppler frequencies in Hz.
        # Frequency resolution is 1 / (N_chirps * T_loop).
        f_doppler = doppler_idx / (N_CHIRPS_LOOPS * T_loop)

        # Compute the first-order correction phasor for TX1 (n=1): one T_chirp delay.
        corr1 = np.exp(-1j * 2 * np.pi * f_doppler * CHIRP_CYCLE_TIME)

        # Compute the second-order correction phasor for TX2 (n=2): two T_chirp delays.
        # corr2 = corr1^2 because the phase error accumulates linearly with TX index.
        corr2 = corr1 ** 2

        # Copy the input frame to avoid in-place modification of the original array.
        out = frame.copy()

        # Apply the first-order correction to the TX1 virtual receivers (indices 4-7).
        # TX0's virtual receivers (indices 0-3) require no correction (n=0).
        # The None dimensions broadcast the 1-D correction vector across frames and range bins.
        out[:, :, [4, 5, 6, 7], :] *= corr1[None, :, None, None]

        # Apply the second-order correction to the TX2 virtual receivers (indices 8-11).
        out[:, :, [8, 9, 10, 11], :] *= corr2[None, :, None, None]

        return out

    @staticmethod
    def fb_spatial_smoothing(R: np.ndarray, L: int) -> np.ndarray:
        """
        Applies Forward-Backward (FB) spatial smoothing to restore the rank of the
        covariance matrix prior to MUSIC decomposition.

        A single snapshot yields only a rank-1 covariance matrix, which collapses the
        noise subspace and makes direct MUSIC application degenerate. Spatial smoothing
        partitions the M x M full-array covariance matrix into K = M - L + 1 overlapping
        L x L sub-array matrices and averages them (forward estimate R_f). The backward
        estimate R_b is derived by conjugating and spatially reversing R_f via the exchange
        matrix J. The FB average (R_f + R_b) / 2 is full-rank even for a single snapshot,
        enabling MUSIC to resolve coherent or correlated sources.

        Parameters
        ----------
        R : ndarray, shape (M, M)
            Full-array spatial covariance matrix (M = total array elements used).
        L : int
            Sub-array length. Typically chosen as L ~ M - 1 to maximise rank while
            maintaining two sub-arrays for averaging.

        Returns
        -------
        ndarray, shape (L, L)
            Forward-backward smoothed covariance matrix.
        """
        # M: full-array aperture (total number of elements passed to this function).
        M = R.shape[0]

        # K: number of overlapping L-element sub-arrays that tile the full aperture.
        K = M - L + 1

        # Accumulate the forward sub-array covariance estimates.
        # Sub-array i uses elements i through i+L-1, corresponding to the L x L principal
        # sub-matrix R[i:i+L, i:i+L].
        Rf = np.zeros((L, L), dtype=complex)
        for i in range(K):
            Rf += R[i:i + L, i:i + L]

        # Normalise by the number of sub-arrays to obtain the averaged forward estimate.
        Rf /= K

        # Construct the L x L exchange (reversal) matrix J.
        # J * A * J spatially reverses the array index ordering of matrix A.
        J = np.fliplr(np.eye(L))

        # Derive the backward covariance estimate by conjugating and spatially reversing R_f.
        # R_b captures the same signals from the opposite array direction, adding an independent
        # rank contribution that further decorrelates coherent sources.
        Rb = J @ Rf.conj() @ J

        # Return the Forward-Backward average; the 0.5 factor normalises the combination.
        return 0.5 * (Rf + Rb)

    @staticmethod
    def music_algorithm(R: np.ndarray, M: int, num_sources: int,
                        scan_angles: np.ndarray) -> np.ndarray:
        """
        Computes the MUSIC (MUltiple SIgnal Classification) pseudo-spectrum for a given
        spatial covariance matrix and a set of candidate scan angles.

        The M x M covariance matrix is decomposed via SVD. The M - num_sources right
        singular vectors corresponding to the smallest singular values span the noise
        subspace, which is orthogonal to the steering vectors of the true sources. The
        MUSIC pseudo-spectrum is the reciprocal of the squared projection of each steering
        vector onto the noise subspace; it peaks sharply at the true directions of arrival.

        P_MUSIC(theta) = 1 / || E_n^H * a(theta) ||^2

        Parameters
        ----------
        R          : ndarray, shape (M, M)
            Spatial covariance matrix (typically forward-backward smoothed).
        M          : int
            Number of array elements represented in R.
        num_sources: int
            Estimated number of signal sources (D < M).
        scan_angles: ndarray, shape (N_angles,)
            Candidate azimuth angles in degrees.

        Returns
        -------
        ndarray, shape (N_angles,)
            MUSIC pseudo-spectrum; peaks identify the estimated directions of arrival.
        """
        # Decompose via SVD; rows of Vh are right singular vectors ordered from the
        # largest to the smallest singular value.
        U, Lambda, Vh = svd(R)

        # Extract the noise subspace basis: rows from index num_sources onward correspond
        # to the M-D smallest singular values. Conjugate-transposing gives an M x (M-D) matrix
        # whose columns span the noise subspace.
        noise_subspace = Vh[num_sources:].conj().T  # shape: (M, M - num_sources)

        # Convert candidate scan angles from degrees to radians.
        angles_rad = np.radians(scan_angles)

        # Build the array steering matrix for a half-wavelength-spaced ULA.
        # Element d at angle theta: phase = pi * d * sin(theta).
        # np.outer generates all (element index, angle) combinations simultaneously.
        steering_matrix = np.exp(1j * np.pi * np.outer(np.arange(M), np.sin(angles_rad)))

        # Evaluate the MUSIC pseudo-spectrum.
        # At a true source angle, the steering vector is orthogonal to the noise subspace
        # (projection norm approaches zero), causing P_music to peak sharply.
        P_music = 1 / (np.linalg.norm(steering_matrix.conj().T @ noise_subspace, axis=1) ** 2)

        return P_music

    def music_per_cell(self, snapshot: np.ndarray, scan_angles: np.ndarray,
                       L: int = 7, K: int = 4):
        """
        Applies 1-D MUSIC to the 8-element azimuth row of a virtual-array snapshot to
        estimate the azimuth angle(s) of arriving signals.

        Only the first 8 elements of the 12-element snapshot are used (virtual indices 0-7),
        corresponding to the contiguous azimuth ULA row at y=0. The elevation row
        (indices 8-11) is deliberately excluded because it does not form a ULA co-linear
        with the azimuth row; including it would corrupt the azimuth steering model.
        Elevation is estimated separately using monopulse (see estimate_elevation).

        With L=7 and 8 azimuth elements, K = M - L + 1 = 2 sub-arrays are averaged in the
        spatial smoothing step, yielding a 7x7 smoothed covariance matrix that supports
        resolution of up to L - 1 = 6 sources. The default K=4 allows detection of up to
        4 simultaneously present signals in a single range-Doppler cell.

        Parameters
        ----------
        snapshot    : ndarray, shape (n_rx,) where n_rx >= 8
            Complex baseband signal vector from all virtual channels for one (range, Doppler) cell.
        scan_angles : ndarray, shape (n_az,)
            Candidate azimuth angles in degrees.
        L           : int, default 7
            Sub-array length for spatial smoothing. Must satisfy K < L <= 8.
        K           : int, default 4
            Estimated number of signal sources in this cell.

        Returns
        -------
        P_music     : ndarray, shape (n_az,)
            MUSIC pseudo-spectrum over the candidate azimuth angles.
        scan_angles : ndarray, shape (n_az,)
            The same scan angle axis, returned for downstream convenience.
        """
        # Extract the 8-element azimuth row (virtual elements 0-7).
        # The elevation row (elements 8-11) is excluded to keep the steering model
        # consistent with a co-linear ULA.
        x = snapshot[:8]  # shape: (8,)

        # Form the rank-1 covariance matrix from the 8-element azimuth snapshot.
        R = np.outer(x, x.conj())  # shape: (8, 8), rank 1

        # Apply forward-backward spatial smoothing to restore the matrix rank.
        # With L=7 and M=8, K_sub = 2 sub-arrays are averaged.
        R_smooth = self.fb_spatial_smoothing(R, L)  # shape: (L, L)

        # Compute the MUSIC pseudo-spectrum on the smoothed covariance matrix.
        P_music = self.music_algorithm(R_smooth, L, K, scan_angles)

        return P_music, scan_angles

    @staticmethod
    def estimate_elevation(snapshot: np.ndarray, az_deg: float) -> float:
        """
        Estimates the elevation angle of a target using monopulse processing on the
        2-row virtual aperture (azimuth row at y=0, elevation row at y=1).

        The elevation row (virtual elements 8-11) shares four column positions (x=2..5)
        with the azimuth row (virtual elements 2-5). At a known azimuth angle, the
        azimuth-direction phase progression across those four columns is compensated by
        multiplying by the conjugate of the azimuth steering vector. The remaining
        inter-row phase difference is caused solely by the elevation angle:

            Delta_phi = 2*pi * (lambda/2) * sin(el) / lambda = pi * sin(el)

        Inverting: sin(el) = Delta_phi / pi  =>  el = arcsin(Delta_phi / pi)

        Parameters
        ----------
        snapshot : ndarray, shape (n_rx,) where n_rx >= 12
            Complex baseband signal vector from all 12 virtual channels.
        az_deg   : float
            Azimuth angle estimate in degrees (from MUSIC azimuth estimation).

        Returns
        -------
        float
            Elevation angle estimate in degrees. Clipped to [-90, +90] via arcsin.
        """
        # Convert the known azimuth angle estimate to radians.
        az = np.deg2rad(az_deg)

        # Compute the azimuth spatial frequency for the 4 shared columns.
        kx = np.pi * np.sin(az)

        # Build the azimuth compensation vector for the 4 shared columns.
        # The negative exponent (conjugate) removes the azimuth phase component from
        # the sum, isolating the inter-row (elevation) phase difference.
        a_az = np.exp(-1j * np.arange(4) * kx)  # shape: (4,)

        # Azimuth-compensated sum for the azimuth row at the 4 overlap positions
        # (virtual elements 2, 3, 4, 5 at absolute x-positions 2, 3, 4, 5).
        row0 = np.sum(snapshot[[2, 3, 4, 5]] * a_az)

        # Azimuth-compensated sum for the elevation row
        # (virtual elements 8, 9, 10, 11 at the same x-positions 2, 3, 4, 5).
        row1 = np.sum(snapshot[[8, 9, 10, 11]] * a_az)

        # Compute the inter-row phase difference via the argument of the cross-correlation.
        # row0 * conj(row1) yields a phasor whose angle equals the inter-row phase difference.
        phase_diff = np.angle(row0 * np.conj(row1))

        # Convert the phase difference to a sine of the elevation angle.
        # With half-wavelength row spacing, Delta_phi = pi * sin(el), so sin(el) = Delta_phi / pi.
        sin_el = phase_diff / np.pi

        # Invert to obtain the elevation angle in degrees.
        # np.clip guards against floating-point values slightly outside [-1, 1]
        # that would cause arcsin to return NaN.
        return np.rad2deg(np.arcsin(np.clip(sin_el, -1, 1)))

    def compute_music_heatmap(self, rd_cube: np.ndarray, frame_idx: int = 0,
                              num_targets: int = 4):
        """
        Computes a 2-D range-azimuth MUSIC pseudo-spectrum heatmap for visualisation and
        algorithm validation.

        For each range bin, the Doppler bin exhibiting the highest integrated power across
        all 12 virtual receivers is selected as the representative snapshot for MUSIC.
        Only the 8-element azimuth row is passed to MUSIC (the elevation row is excluded).
        The resulting heatmap shows the distribution of target energy across the range-azimuth
        plane and validates that detected targets appear at physically expected angular positions.

        Parameters
        ----------
        rd_cube    : ndarray, shape (frames, doppler, rx_virtual, range)
            Phase-corrected range-Doppler cube.
        frame_idx  : int, default 0
            Index of the frame to process.
        num_targets: int, default 4
            Estimated number of sources; passed to MUSIC as K.

        Returns
        -------
        heatmap    : ndarray, shape (n_range, n_az)
            MUSIC pseudo-spectrum value for every (range bin, azimuth angle) pair.
        scan_angles: ndarray, shape (n_az,)
            Azimuth angle axis in degrees, spanning +-azimuth_fov_degrees at 0.1-degree steps.
        """
        # Select the target frame and unpack its dimensional extents.
        frame = rd_cube[frame_idx]          # shape: (doppler, rx_virtual, range)
        n_doppler, n_rx, n_range = frame.shape

        # Compute the per-Doppler-bin power integrated across all 12 virtual receivers.
        # The (doppler, range) power map is used only to identify the peak-SNR Doppler
        # snapshot at each range gate.
        doppler_power = (np.abs(frame) ** 2).sum(axis=1)   # shape: (doppler, range)

        # For each range bin, find the Doppler bin index carrying maximum integrated power.
        # This snapshot provides the best SNR for MUSIC angle estimation at that range gate.
        peak_dop = np.argmax(doppler_power, axis=0)         # shape: (range,)

        # Define the azimuth scan grid spanning +-azimuth_fov_degrees at 0.1-degree steps.
        scan_angles = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)

        # Allocate the output heatmap; each row corresponds to one range bin.
        heatmap = np.zeros((n_range, len(scan_angles)))

        # Iterate over every range bin and compute the MUSIC azimuth pseudo-spectrum.
        for r in range(n_range):
            # Extract the 12-element spatial snapshot at the peak Doppler bin.
            x = frame[peak_dop[r], :, r]   # shape: (rx_virtual,)

            # Run 1-D MUSIC on the azimuth row (elements 0-7); K = num_targets.
            P, _ = self.music_per_cell(x, scan_angles, K=num_targets)

            # Store the pseudo-spectrum as one row of the heatmap.
            heatmap[r] = P

        return heatmap, scan_angles

    @staticmethod
    def cfar_detection(data: np.ndarray, guard_doppler, guard_range,
                       train_doppler, train_range, k_rank, pfa):
        """
        Applies 2-D OS-CFAR (Ordered-Statistics CFAR) detection to the range-Doppler
        power map and returns a boolean detection mask with the per-cell threshold map.

        OS-CFAR estimates the local noise power by sorting the N training cell values and
        selecting the k-th order statistic as the noise estimate. This rank-based approach
        is robust at clutter boundaries and near the zero-Doppler ridge, where a minority
        of training cells may carry elevated power that would bias a mean-based (CA-CFAR)
        estimate. Guard cells surrounding the CUT are excluded from the training window to
        prevent target energy from leaking into the noise estimate.

        Parameters
        ----------
        data          : ndarray, shape (frames, doppler, range)
            Range-Doppler integrated power map.
        guard_doppler : int
            Guard cells per side along the Doppler axis (excluded from training window).
        guard_range   : int
            Guard cells per side along the range axis.
        train_doppler : int
            Training cells per side along the Doppler axis.
        train_range   : int
            Training cells per side along the range axis.
        k_rank        : int or None
            Rank index for OS-CFAR (0 = smallest; N-1 = largest training cell).
            Pass None to use the implementation default (typically ~0.75 * N).
        pfa           : float
            Target probability of false alarm; controls the threshold multiplier alpha.

        Returns
        -------
        detection_mask : ndarray, shape (frames, doppler, range), dtype bool
            True at every cell whose power exceeds the adaptive detection threshold.
        threshold_map  : ndarray, shape (frames, doppler, range)
            Absolute detection threshold value computed for each cell.
        """
        # Delegate to the OS-CFAR implementation for the full 3-D data cube.
        return os_cfar_on_cube(data, guard_doppler, guard_range,
                               train_doppler, train_range, k_rank, pfa)

    def point_cloud_from_rd(self, rd_cube: np.ndarray,
                            rd_det: np.ndarray, K: int = 4) -> List[np.ndarray]:
        """
        Runs 1-D MUSIC azimuth estimation and monopulse elevation estimation on every
        CFAR-detected range-Doppler cell to assemble a full 5-attribute per-frame point cloud.

        For each detection cell, the full 12-element virtual-array snapshot is used:
            - MUSIC operates on elements 0-7 (azimuth row) to estimate the azimuth angle.
            - Monopulse operates on the 4-column overlap between rows (elements 2-5 vs 8-11)
              to estimate elevation at the known azimuth.
        Detections whose elevation magnitude exceeds elevation_fov_degrees are rejected as
        physically implausible (likely grating lobes or noise-induced outliers).

        Each output point carries five attributes:
            range_m      : physical slant range in metres.
            velocity     : signed radial velocity in m/s (positive = receding, negative = approaching).
            azimuth_deg  : estimated azimuth angle in degrees.
            elevation_deg: estimated elevation angle in degrees.
            power_dBFS   : total received power of the azimuth-row snapshot in dBFS.

        Parameters
        ----------
        rd_cube : ndarray, shape (frames, doppler, rx_virtual, range)
            Phase-corrected range-Doppler cube (all 12 virtual channels).
        rd_det  : ndarray, shape (frames, doppler, range), dtype bool
            CFAR detection mask.
        K       : int, default 4
            Maximum number of azimuth peaks (MUSIC) to extract per detection cell.

        Returns
        -------
        list of ndarray, length n_frames
            Each element is an (n_pts, 5) float array with columns
            [range_m, velocity_m_s, azimuth_deg, elevation_deg, power_dBFS].
            Frames with no valid detections return an empty (0, 5) array.
        """
        # Unpack frame and Doppler dimension counts from the cube shape.
        n_frames, n_doppler, _, _ = rd_cube.shape

        # Compute the inter-TX loop time for the 3-TX configuration.
        T_loop = self.N_TX * CHIRP_CYCLE_TIME

        # Construct the signed (centred) Doppler bin index array.
        doppler_idx = np.arange(n_doppler) - n_doppler // 2

        # Convert Doppler indices to signed radial velocities in m/s.
        # The radar velocity equation: v = (bin_index * lambda) / (2 * N_doppler * T_loop).
        v_axis = doppler_idx * WAVELENGTH / (2 * n_doppler * T_loop)

        # Define the azimuth scan grid used by MUSIC.
        scan_angles = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)

        out = []
        for f in range(n_frames):
            # Extract the CFAR detection mask for this frame.
            det = rd_det[f]  # shape: (doppler, range), dtype bool

            # Retrieve the (Doppler, range) index pairs of all declared detections.
            d_idx, r_idx = np.where(det)

            pts = []
            for d, r in zip(d_idx, r_idx):
                # Extract the full 12-element virtual-array snapshot for this cell.
                # x encodes the spatial phase pattern needed by both MUSIC and monopulse.
                x = rd_cube[f, d, :, r]  # shape: (rx_virtual,)

                # Run 1-D MUSIC on the 8-element azimuth row to obtain the pseudo-spectrum.
                P, scan = self.music_per_cell(x, scan_angles, K=K)

                # Convert to dB for prominence-based peak detection.
                # The offset 1e-12 prevents log(0) for near-zero spectrum values.
                P_db = 10 * np.log10(P + 1e-12)

                # Find local maxima with at least 6 dB prominence to suppress noise
                # ripple and MUSIC sidelobe artefacts.
                peak_idx, _ = find_peaks(P_db, prominence=6)

                # Rank detected peaks by linear pseudo-spectrum amplitude (descending)
                # and retain at most K (one per expected simultaneous source).
                order = np.argsort(P[peak_idx])[::-1][:K]

                # Compute the received power from the azimuth row only (elements 0-7).
                # Using only the azimuth row keeps the power estimate consistent with the
                # 8-element ULA model used for MUSIC.
                power_db = 10 * np.log10(np.sum(np.abs(x[:8]) ** 2) + 1e-12)

                for pk in peak_idx[order]:
                    # Extract the azimuth angle estimate for this MUSIC peak.
                    az = scan[pk]

                    # Estimate elevation via monopulse at the known azimuth angle.
                    el = self.estimate_elevation(x, az)

                    # Reject detections outside the physical elevation FOV.
                    # This removes grating-lobe ambiguities and noise-induced outliers.
                    if abs(el) > self.elevation_fov_degrees:
                        continue

                    # Append the valid (range, velocity, azimuth, elevation, power) tuple.
                    pts.append((r * RANGE_RESOLUTION, v_axis[d], az, el, power_db))

            # Convert the list of tuples to a NumPy array; empty (0, 5) if no valid detections.
            out.append(np.array(pts) if pts else np.empty((0, 5)))

        return out

    def point_cloud(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Converts per-frame spherical radar detections
        (range_m, velocity_m_s, azimuth_deg, elevation_deg, power_dBFS)
        into Cartesian 3D coordinates (x, y, z) augmented with all original attributes.

        The coordinate convention is:
            x = r * cos(el) * sin(az)   -> lateral (cross-range) axis, positive to the right
            y = r * cos(el) * cos(az)   -> longitudinal (down-range) axis, positive forward
            z = r * sin(el)             -> vertical axis, positive upward

        The cos(el) factor in x and y accounts for elevation foreshortening: the horizontal
        ground-plane range of a target at slant range r and elevation el is r * cos(el).

        Parameters
        ----------
        data : list of ndarray, each of shape (N, 5)
            Per-frame detection arrays with columns
            [range_m, velocity_m_s, azimuth_deg, elevation_deg, power_dBFS].

        Returns
        -------
        list of ndarray, each of shape (N, 8)
            Per-frame Cartesian point clouds with columns
            [x_m, y_m, z_m, range_m, velocity_m_s, azimuth_deg, elevation_deg, power_dBFS].
            Frames with no detections return an empty (0, 8) array.
        """
        def to_xyz_full(detections: np.ndarray) -> np.ndarray:
            """Projects a single frame's spherical detections to Cartesian 3D coordinates."""
            # Unpack the five detection attributes from the input array columns.
            r_m      = detections[:, 0]   # slant range in metres
            vel      = detections[:, 1]   # radial velocity in m/s
            az_deg   = detections[:, 2]   # azimuth angle in degrees
            el_deg   = detections[:, 3]   # elevation angle in degrees
            power_db = detections[:, 4]   # signal power in dBFS

            # Convert angular measurements to radians for trigonometric computation.
            az_rad = np.deg2rad(az_deg)
            el_rad = np.deg2rad(el_deg)

            # Project spherical coordinates (r, az, el) into the Cartesian frame.
            x = r_m * np.cos(el_rad) * np.sin(az_rad)   # lateral displacement
            y = r_m * np.cos(el_rad) * np.cos(az_rad)   # longitudinal displacement
            z = r_m * np.sin(el_rad)                      # vertical displacement

            # Stack all eight output columns (Cartesian + original attributes) into one array.
            return np.stack([x, y, z, r_m, vel, az_deg, el_deg, power_db], axis=1)

        # Process each frame; preserve the empty (0, 8) structure for frames with no detections.
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
        Runs MUSIC azimuth estimation on every CFAR-detected range-Doppler cell and
        assembles a per-frame azimuth-only point cloud for intermediate visualisation.

        This method mirrors the output format of the azimuth-only pipeline
        (PointCloud1DMUSIC.point_cloud_from_rd_azimuth), allowing the 3-TX processing
        chain to be validated at the azimuth stage independently of elevation estimation.
        Each output point carries four attributes: range, velocity, azimuth, and power.

        Parameters
        ----------
        rd_cube : ndarray, shape (frames, doppler, rx_virtual, range)
            Phase-corrected range-Doppler cube (all 12 virtual channels).
        rd_det  : ndarray, shape (frames, doppler, range), dtype bool
            CFAR detection mask.
        K       : int, default 3
            Maximum number of azimuth peaks (MUSIC) to extract per detection cell.

        Returns
        -------
        list of ndarray, length n_frames
            Each element is an (n_pts, 4) float array with columns
            [range_m, velocity_m_s, azimuth_deg, power_dBFS].
            Frames with no detections return an empty (0, 4) array.
        """
        # Unpack frame and Doppler dimension counts.
        n_frames, n_doppler, _, _ = rd_cube.shape

        # Use a 2-TX inter-loop time for this intermediate azimuth-only output.
        # This keeps the velocity axis consistent with the 2-TX azimuth-only pipeline
        # convention for direct comparison during validation.
        T_loop = 2 * CHIRP_CYCLE_TIME

        # Construct the signed (centred) Doppler bin index array matching the fftshift convention.
        # Velocity axis, matching the fftshift'd Doppler bins.
        doppler_idx = np.arange(n_doppler) - n_doppler // 2

        # Convert Doppler indices to signed radial velocities in m/s.
        v_axis = doppler_idx * WAVELENGTH / (2 * n_doppler * T_loop)

        # Define the azimuth scan grid used by MUSIC.
        scan_angles = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)

        out = []
        for f in range(n_frames):
            # Extract the CFAR detection mask for this frame.
            det = rd_det[f]  # shape: (doppler, range), dtype bool

            # Retrieve (Doppler, range) index pairs for all declared detections.
            d_idx, r_idx = np.where(det)

            pts = []
            for d, r in zip(d_idx, r_idx):
                # Extract the 12-element virtual-array snapshot for this cell.
                x = rd_cube[f, d, :, r]  # shape: (rx_virtual,)

                # Run 1-D MUSIC on the azimuth row (elements 0-7) to obtain the pseudo-spectrum.
                P, scan = self.music_per_cell(x, scan_angles, K=K)

                # Convert to dB for prominence-based peak detection.
                P_db = 10 * np.log10(P + 1e-12)

                # Identify local maxima with a minimum 6 dB prominence threshold.
                peak_idx, _ = find_peaks(P_db, prominence=6)

                # Retain at most K strongest peaks, ranked by linear pseudo-spectrum amplitude.
                # take up to K strongest
                order = np.argsort(P[peak_idx])[::-1][:K]

                # Compute the total received power from the full 12-element snapshot in dBFS.
                power_db = 10 * np.log10(np.sum(np.abs(x) ** 2) + 1e-12)

                # Append one 4-tuple per retained MUSIC peak.
                for pk in peak_idx[order]:
                    pts.append((r * RANGE_RESOLUTION, v_axis[d], scan[pk], power_db))

            # Convert the list of tuples to a NumPy array; empty (0, 4) if no detections.
            out.append(np.array(pts) if pts else np.empty((0, 4)))

        return out


if __name__ == '__main__':
    adc_data = parse_mmwave_studio_adc(RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, 3, N_FRAMES)

    p = PointCloud2DMUSIC()
    p.run(adc_data, static_removal=False, phase_shift=True)
