import math
from typing import List, Optional

import numpy as np
from scipy.signal import find_peaks

from parser.parser_dca1000 import parse_mmwave_studio_adc
from plots.range import plot_range_fft
from utils.config import RAW_ADC_PATH, N_FRAMES, RX_CHANNELS, ADC_SAMPLES, \
    N_CHIRPS_LOOPS, RANGE_RESOLUTION, CHIRP_CYCLE_TIME, TX_CHANNELS, WAVELENGTH
from plots.azimuth_spectrum import plot_azimuth_spectrum
from plots.range_doppler import plot_range_doppler
from plots.point_cloud import plot_point_cloud_rd
from plots.music_spectrum import plot_music_range_azimuth_heatmap
from scipy.linalg import svd
from cfar.cfar_2d import peak_group, os_cfar_on_cube, ca_cfar_on_cube


class PointCloud1DMUSIC:
    """
    End-to-end mmWave radar signal processing pipeline that produces a 3D point cloud
    (range, velocity, azimuth) from raw ADC samples captured with a TDM-MIMO radar.

    The processing chain is as follows:
        Range FFT  ->  Static Clutter Removal  ->  Doppler FFT
        ->  TDM-MIMO Phase Correction  ->  2-D CFAR Detection
        ->  MUSIC Angle Estimation  ->  Point Cloud Assembly

    The virtual array is formed by two transmitters and four receivers arranged such that
    the 8 virtual elements are co-linear along the azimuth axis at half-wavelength spacing,
    enabling 1-D MUSIC-based angle-of-arrival estimation.

    Transmitter-to-port mapping (as configured in mmWave Studio):
        TX0 (First Checkbox in mmWave Studio)  -> Physical TX1 -> Azimuth plane
        TX1 (Second Checkbox in mmWave Studio) -> Physical TX3 -> Azimuth plane
    """

    # Maximum azimuth half-angle in degrees; the MUSIC pseudo-spectrum is scanned
    # symmetrically from -azimuth_fov_degrees to +azimuth_fov_degrees.
    azimuth_fov_degrees = 60

    # Virtual array element positions expressed in half-wavelength units.
    # Column 0 is the x-coordinate (along the azimuth axis); column 1 is the
    # y-coordinate (along the elevation axis, always zero for this planar array).
    # The 8 elements span indices 0 through 7 at unit spacing (0.5 lambda between
    # consecutive elements), satisfying the spatial Nyquist criterion and preventing
    # aliasing up to +-90 degrees in azimuth.
    pos = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0]
    ])

    def run(self, adc_data: np.ndarray, static_removal: bool = True, phase_shift: bool = True,
            mask_zero_doppler: bool = True, num_targets: int = 1,
            peak_group_neighborhood: int = 0, range_max: Optional[int] = None):
        """
        Executes the full radar signal processing pipeline on a batch of raw ADC frames.

        The pipeline converts raw complex ADC samples into a 3D point cloud by sequentially
        applying: Range FFT, optional static clutter removal, Doppler FFT, optional
        TDM-MIMO phase correction, 2-D CFAR detection, and MUSIC angle estimation.
        Intermediate results are visualized through dedicated plotting routines at key stages.

        Parameters
        ----------
        adc_data               : ndarray, shape (frames, chirps, rx_virtual, adc_samples)
            Raw complex ADC data as captured by the radar front-end.
        static_removal         : bool, default True
            If True, subtracts the intra-frame chirp mean to suppress static clutter.
        phase_shift            : bool, default True
            If True, applies TDM-MIMO inter-TX phase correction after the Doppler FFT.
        mask_zero_doppler      : bool, default True
            If True, suppresses the zero-Doppler bins in the detection mask to exclude
            residual static returns that survived clutter removal.
        num_targets            : int, default 1
            Estimated number of simultaneous signal sources; passed to MUSIC as K.
        peak_group_neighborhood: int, default 0
            If greater than zero, performs peak grouping with this neighborhood radius
            to reduce each detection cluster to a single representative cell.
        range_max              : float or None, default None
            Maximum range of interest in metres. If provided and shorter than the
            maximum unambiguous range, the range-FFT output is cropped accordingly.
        """
        # Assign the input ADC data to the working buffer.
        frame = adc_data

        # --- Stage 1: Range FFT ---
        # Apply the first FFT (Range FFT) along the ADC sample (fast-time) axis.
        # Each beat-frequency tone in the time-domain ADC record maps to a discrete
        # range bin whose index is proportional to the round-trip delay of the target.
        frame = self.range_fft(frame, True)

        # --- Stage 2: Range Gate Selection ---
        # By default, all ADC_SAMPLES range bins are retained.
        max_bin = ADC_SAMPLES

        # If the caller specifies a maximum range of interest shorter than the full
        # unambiguous range, compute the corresponding bin index and crop the array.
        # math.ceil ensures that a target located at exactly range_max falls within the window.
        if range_max is not None and range_max < ADC_SAMPLES * RANGE_RESOLUTION:
            max_bin = math.ceil(range_max / RANGE_RESOLUTION)
            frame = frame[:, :, :, :max_bin]

        # Plot the Range FFT magnitude spectrum before clutter removal to show the raw
        # range profile. When plotted against the range axis, the magnitude spectrum
        # shows a sharp peak at the range corresponding to the object that reflects the
        # chirp signal back to the radar sensor.
        plot_range_fft(frame[0, 0, 0, :], max_bin, RANGE_RESOLUTION)

        # --- Stage 3: Static Clutter Removal ---
        # In an indoor scene, the radar signal bounces off walls, floor, ceiling,
        # furniture, and the radar's own mounting structure and the
        # surrounding static environment. These returns produce large, stationary peaks
        # in the range profile that can mask weaker moving-target returns.
        #
        # A running average (or intra-frame mean across chirps) estimates, for each
        # range-channel cell, the average signal component that is constant across chirps
        # (i.e., the static background). Subtracting this estimate from every chirp
        # suppresses DC and near-DC returns, leaving only the time-varying (moving)
        # components.
        if static_removal:
            frame = self.static_clutter_removal(frame)

            # Re-plot the range profile after clutter removal to confirm that the static
            # peaks have been reduced and the moving-target peak is more prominent.
            plot_range_fft(frame[0, 0, 0, :], max_bin, RANGE_RESOLUTION)

        # --- Stage 4: Doppler FFT ---
        # Apply the second FFT (Doppler FFT) across the chirp (slow-time) axis.
        # A moving target introduces a linearly increasing inter-chirp phase shift
        # proportional to its radial velocity; the Doppler FFT maps this phase
        # progression to a unique velocity bin.
        frame = self.doppler_fft(frame, True)

        # --- Stage 5: TDM-MIMO Phase Correction ---
        # In Time-Division Multiplexed MIMO, each TX antenna fires in a dedicated time slot
        # within the chirp cycle. For a moving target, the range changes between consecutive
        # TX firings, causing additional phase accumulation proportional to the target's
        # radial velocity and the inter-TX delay. Without correction, this phase error
        # appears as a fictitious angle offset in the virtual-array steering vector,
        # displacing the apparent direction of arrival from the true physical angle.
        if phase_shift:
            frame = self.phase_correction(frame)

        # --- Stage 6: Range-Doppler Power Map ---
        # Compute the incoherent power sum across all virtual receivers.
        # Summing the squared magnitudes across the receiver axis improves SNR via
        # non-coherent integration gain, producing a 2-D (Doppler x range) power map
        # suitable for CFAR thresholding.
        rd_power = (np.abs(frame) ** 2).sum(axis=2)  # shape: (frames, doppler, range)

        # Plot the range-Doppler heatmap for visual inspection.
        # Convention: a positive velocity indicates the object moving away from the radar,
        # and a negative velocity indicates the object moving towards the radar.
        # The object reflecting the chirp signals back to the radar can be distinguished
        # with a sharp peak at the coordinate corresponding to its range and velocity values.
        plot_range_doppler(rd_power, frame_idx=0)

        # --- Stage 7: 2-D CFAR Detection ---
        # Apply a Constant False Alarm Rate detector to identify target-bearing cells in
        # the range-Doppler power map. Instead of using a fixed detection threshold, CFAR
        # sets the threshold dynamically based on an estimate of the local background noise,
        # maintaining an approximately constant probability of false alarm regardless of the
        # noise level.
        #
        # CA-CFAR vs. OS-CFAR at the zero-Doppler ridge:
        #   CA-CFAR will struggle at the ridge edges. As the window slides off the zero-Doppler
        #   line, training cells are split between high-power (still on ridge) and low-power
        #   (off ridge) cells. The mean sits between, and the result is either false alarms
        #   just past the edge (threshold too low) or missed detections at the edge itself
        #   (threshold too high). OS-CFAR's k-th order statistic handles this better: if
        #   k = 0.75*N, then as long as fewer than 25% of the training cells are on the ridge,
        #   the rank-k value reflects the off-ridge noise floor. The transition is sharper and
        #   more correct. In summary: CA-CFAR takes the mean; OS-CFAR sorts and takes the k-th smallest.
        #
        # Tight configuration (guard=2, train=4, pfa=1e-7):
        #   Small window, approximately 144 training cells, highly localized noise estimate.
        #   Threshold multiplier alpha ~ 16, corresponding to ~12 dB above the local mean.
        #   Picks up the target cleanly but tightly: only cells that directly exceed the local
        #   floor. Results in few detection points per target.
        #
        # Wide configuration (guard=8, train=32, pfa=1e-16):
        #   Large window, more than 4000 training cells. The noise estimate approximates the
        #   global mean of the clutter-removed map. Threshold multiplier alpha ~ 37,
        #   corresponding to ~16 dB above the global mean. After static removal, the map is
        #   dominated by thermal noise everywhere, making the threshold approximately constant
        #   across the entire range-Doppler plane.
        #
        # Why the wide configuration produces more detection points around the moving target:
        #   After static removal, the target is the only bright structure in the map. With the
        #   tight configuration, training cells near the target include energy from the target's
        #   main lobe and sidelobes (the Hann window places the first sidelobe at ~-32 dB, but
        #   extended target energy spreads further). The guard region (central 5x5) excludes
        #   the peak, but train_range=4 means training cells extend only 4-6 bins outside the
        #   guard, exactly where the target's sidelobes still reside. The local noise estimate
        #   is therefore inflated by the target's own skirt, raising the threshold near the
        #   target and suppressing weaker peripheral bins. Only the bright core survives.
        #   The wide configuration places the guard region 17x17 around the CUT, large enough
        #   to contain all significant target energy inside the guard. Training cells are far
        #   enough away to sample pure thermal noise. The threshold near the target therefore
        #   reflects thermal noise, not skirt-inflated noise. Peripheral bins that were above
        #   thermal noise but below the tight configuration's inflated threshold now survive,
        #   producing more detection points per target. The aggressive pfa=1e-16 compensates
        #   for the large N (so alpha stays sensible) and further lowers the threshold relative
        #   to what pfa=1e-7 with the same large window would give.
        #
        # When to prefer each configuration:
        #   Wide config is preferable when the map is genuinely flat (clean static removal, no
        #   residual clutter), there are one or few well-separated targets, and downstream
        #   processing benefits from target shape or cluster extent (e.g., multiple snapshots
        #   per target for MUSIC). Wide config degrades when two targets are close together
        #   (one target's guards overlap the other, biasing the global threshold) or when
        #   residual clutter patches remain (the global mean is locally incorrect, causing
        #   false alarms or missed detections in different regions).
        #   Tight config with peak grouping is preferred for single-point-per-target tracking.
        rd_det, _ = self.cfar_detection(rd_power,
                                        guard_doppler=1, guard_range=2,
                                        train_doppler=2, train_range=4,
                                        k_rank=None, pfa=1e-7
                                        )

        # --- Stage 8: Edge Range Bin Suppression ---
        # Blank the first and last 8 range bins in the detection mask.
        # These bins are dominated by electronic artifacts such as DC offset and
        # receiver-chain transients rather than genuine reflected target energy.
        rd_det[:, :, :8] = False   # suppress the near-range edge bins
        rd_det[:, :, -8:] = False  # suppress the far-range edge bins

        # --- Stage 9: Optional Peak Grouping ---
        # If a neighborhood radius is specified, replace each detection cluster with a
        # single representative cell at the cluster's power-weighted center. This is useful
        # when a single physical target illuminates multiple adjacent range-Doppler bins and
        # only one detection per target is desired (e.g., for downstream tracking filters).
        if peak_group_neighborhood > 0:
            for f in range(rd_det.shape[0]):
                rd_det[f] = peak_group(rd_power[f], rd_det[f], neighborhood=peak_group_neighborhood)

        # --- Stage 10: Zero-Doppler Masking ---
        # Suppress the three central Doppler bins (the zero-Doppler line and its immediate
        # neighbors) to remove residual static returns that survived clutter removal.
        # This step is appropriate when the targets of interest are moving, since stationary
        # objects should not produce valid detections after clutter removal has been applied.
        if mask_zero_doppler:
            zero_dop = rd_det.shape[1] // 2  # index of the DC (zero-velocity) Doppler bin
            rd_det[:, zero_dop - 1:zero_dop + 2, :] = False

        # --- Stage 11: MUSIC Range-Azimuth Heatmap ---
        # Compute and display the MUSIC pseudo-spectrum heatmap over the full range-azimuth
        # plane to validate that targets appear at expected angular positions and to observe
        # the effect of forward-backward spatial smoothing. The estimated number of targets
        # must be strictly less than the smoothed sub-array dimension L (default L=4);
        # if more targets are expected simultaneously, L must be increased accordingly.
        heatmap, _ = self.compute_music_heatmap(frame, 0, num_targets)
        plot_music_range_azimuth_heatmap(heatmap, self.azimuth_fov_degrees, frame_idx=0)

        # --- Stage 12: Point Cloud Assembly ---
        # Run MUSIC on each CFAR-detected (range, Doppler) cell to estimate the azimuth
        # angle of arrival and assemble the (range, velocity, azimuth, power) point cloud.
        clouds = self.point_cloud_from_rd_azimuth(frame, rd_det, num_targets)
        plot_point_cloud_rd(clouds, frame_idx=0)

    @staticmethod
    def static_clutter_removal(data: np.ndarray):
        """
        Removes static clutter by subtracting the intra-frame mean across the slow-time
        (chirp) axis.

        Static objects, such as walls, floor, ceiling, furniture, and the radar's own mounting
        structure, return nearly identical baseband signals on every chirp within a frame,
        because their range and reflectivity do not change appreciably over the frame duration
        (on the order of milliseconds). Computing the mean of the received signal along the
        chirp axis therefore produces a clean estimate of the static background contribution
        for each (receiver channel, range bin) cell. Subtracting this estimate from every
        chirp snapshot leaves only the time-varying component, i.e., the signal attributed
        to moving targets.

        This operation is equivalent to a single-tap high-pass filter applied along the
        slow-time (chirp) axis and is the simplest form of Moving Target Indication (MTI).

        Input:  data of shape (frames, chirps, rx_virtual, range)
        Output: data of shape (frames, chirps, rx_virtual, range)
        """
        # Compute the mean signal across all chirps within each frame, independently for
        # each receiver channel and range bin. keepdims=True preserves the chirp dimension
        # so that NumPy broadcasting subtracts the mean correctly from every chirp snapshot
        # without requiring an explicit reshape.
        mean_frames = data.mean(axis=1, keepdims=True)

        # Subtract the static background estimate from every chirp snapshot.
        # After subtraction, the result is zero-mean along the chirp axis, effectively
        # suppressing the DC (stationary-target) component of each range-receiver cell.
        return data - mean_frames

    @staticmethod
    def range_fft(data: np.ndarray, use_window: bool):
        """
        Applies the Range FFT (fast-time FFT) to convert raw ADC samples into the range domain.

        Each received chirp is a linear frequency-modulated (LFM) waveform that, after
        down-conversion and mixing with the transmitted replica, produces a sinusoidal beat
        signal whose instantaneous frequency is proportional to the round-trip delay of the
        reflected wavefront, i.e., to the target range. Taking the FFT of the ADC sample
        sequence therefore maps each target to a distinct range bin, with the bin index
        proportional to the target's distance from the radar.

        A Hanning window is optionally applied before the FFT. Windowing tapers
        the endpoints of the time-domain record smoothly to zero, which reduces spectral
        leakage (the smearing of energy from a strong target into adjacent range bins) at
        the cost of a moderate reduction in range resolution (approximately double the
        rectangular-window equivalent, because the width of the main lobe of the windows
        is twice as large as the rectangular window).

        Input:  data of shape (frames, chirps, rx_virtual, adc_samples)
        Output: data of shape (frames, chirps, rx_virtual, range)
        """
        if use_window:
            # Construct a Hanning window of length equal to the number of ADC samples.
            # Coefficients taper from zero at both endpoints to one at the center.
            hanning_window = np.hanning(data.shape[-1])

            # Multiply every ADC sample sequence by the window coefficients element-wise.
            # NumPy broadcasting applies the 1-D window along the last axis (ADC samples)
            # simultaneously across all frames, chirps, and receiver channels.
            data = data * hanning_window

        # Apply the DFT along the last axis (fast time / ADC samples).
        # Each complex output bin corresponds to one range gate; the bin magnitude is
        # proportional to the amplitude of the reflected signal from that range.
        return np.fft.fft(data, axis=-1)

    @staticmethod
    def doppler_fft(data: np.ndarray, use_window: bool):
        """
        Applies the Doppler FFT (slow-time FFT) to convert inter-chirp phase variations
        into the velocity (Doppler) domain.

        A moving target introduces a linearly increasing phase shift from chirp to chirp
        within a frame, because its range, and therefore the beat-signal phase, changes
        between successive chirp firings. The rate of this inter-chirp phase progression is
        proportional to the target's radial velocity. Taking the FFT across the chirp index
        maps each velocity component to a unique Doppler bin. An fftshift operation reorders
        the output so that zero Doppler (stationary targets) is centred on the axis, with
        negative velocities (approaching targets) on the left and positive velocities
        (receding targets) on the right.

        A Hanning window applied along the chirp axis optionally reduces Doppler sidelobes
        at the expense of a small broadening of the velocity resolution cell.

        Input:  data of shape (frames, chirps, rx_virtual, range)
        Output: data of shape (frames, doppler, rx_virtual, range)
        """
        if use_window:
            # Transpose to bring the chirp axis (axis 1) to the last position so that the
            # Hanning window can be applied via broadcasting along the last axis.
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
        # Each complex output bin corresponds to one Doppler (velocity) bin.
        doppler_fft = np.fft.fft(data, axis=1)

        # Shift the zero-frequency (zero-velocity) component to the center of the Doppler
        # axis, converting the default [0, ..., N-1] bin ordering to the symmetric
        # [-N/2, ..., N/2-1] convention used throughout subsequent processing.
        return np.fft.fftshift(doppler_fft, axes=1)

    @staticmethod
    def phase_correction(frame: np.ndarray):
        """
        Compensates for the inter-transmitter phase ramp introduced by target motion in
        Time-Division Multiplexing (TDM) MIMO radar.

        In a TDM-MIMO arrangement, each transmit antenna fires in a dedicated time slot
        within the chirp cycle. A moving target accumulates an additional phase between
        consecutive TX firings because its range changes during the inter-TX delay interval
        T_chirp. Without correction, this velocity-induced phase error manifests as a
        fictitious angular offset in the virtual-array steering vector, displacing the
        apparent angle of arrival away from the true physical direction of the target.

        The correction phasor for a virtual receiver associated with TX index n is:
            phi_n(k) = exp(-j * 2*pi * f_d[k] * n * T_chirp)
        where f_d[k] is the Doppler frequency of the k-th Doppler bin and T_chirp is the
        chirp cycle duration. Because TX0 fires first (n=0), its virtual receivers
        (indices 0-3) carry no correction term. TX1 fires one chirp cycle later (n=1), so
        its virtual receivers (indices 4-7) are multiplied by the correction phasor.

        Input:  data of shape (frames, doppler, rx_virtual, range)
        Output: data of shape (frames, doppler_corrected, rx_virtual, range)
        """
        # Compute the total inter-TX delay: the number of TX channels multiplied by the
        # chirp cycle duration gives the time elapsed between successive transmitter firings.
        T_loop = TX_CHANNELS * CHIRP_CYCLE_TIME

        # Construct the signed (centred) Doppler bin index array to match the fftshift
        # convention applied during the Doppler FFT. Subtracting N_CHIRPS_LOOPS // 2
        # maps bin 0 to -N/2 and bin N-1 to N/2-1.
        doppler_idx = np.arange(N_CHIRPS_LOOPS) - N_CHIRPS_LOOPS // 2

        # Convert Doppler bin indices to physical Doppler frequencies in Hz.
        # The frequency resolution is 1 / (N_chirps * T_loop).
        f_doppler = doppler_idx / (N_CHIRPS_LOOPS * T_loop)

        # Compute the per-Doppler-bin correction phasor for the second transmitter (TX1, n=1).
        # The phasor removes the additional phase that a target moving at the velocity
        # corresponding to each Doppler bin accumulates over one chirp cycle interval.
        correction = np.exp(-1j * 2 * np.pi * f_doppler * CHIRP_CYCLE_TIME)

        # Copy the input frame to avoid modifying the original array in place.
        out = frame.copy()

        # Apply the correction exclusively to virtual receivers associated with TX1 (indices 4-7).
        # TX0's virtual receivers (indices 0-3) have no inter-TX delay (n=0), so no correction
        # is needed. The None dimensions broadcast the 1-D correction vector across frames
        # and range bins.
        out[:, :, [4, 5, 6, 7], :] *= correction[None, :, None, None]

        return out

    @staticmethod
    def fb_spatial_smoothing(R, L):
        """
        Applies Forward-Backward (FB) spatial smoothing to decorrelate coherent sources and
        restore the rank of the covariance matrix prior to MUSIC decomposition.

        Standard MUSIC assumes that the M x M spatial covariance matrix has exactly D large
        eigenvalues (one per source) and M-D small noise eigenvalues. When sources are fully
        coherent (e.g., multipath reflections of the same transmitter), their contributions
        add constructively in the covariance matrix, causing the rank to drop below D and
        making signal/noise subspace separation impossible.

        Forward spatial smoothing partitions the M x M full-array covariance matrix R into
        K = M - L + 1 overlapping L x L sub-array matrices and averages them to form the
        forward estimate R_f. This average is full-rank even when the input signals are
        coherent, provided L is chosen appropriately (typically L ~ M/2 to balance aperture
        loss against rank restoration).

        Backward smoothing derives R_b by conjugating and spatially reversing R_f via the
        L x L exchange matrix J (ones on the anti-diagonal). Averaging R_f and R_b as the
        final FB estimate further improves rank restoration and reduces estimation variance.

        Parameters
        ----------
        R : ndarray, shape (M, M)
            Full-array spatial covariance matrix (M = total number of virtual array elements).
        L : int
            Sub-array length (smoothed aperture). Must satisfy 1 <= L <= M.

        Returns
        -------
        ndarray, shape (L, L)
            Forward-backward smoothed covariance matrix.
        """
        # M: full-array aperture (total number of virtual elements).
        M = R.shape[0]

        # K: number of overlapping L-element sub-arrays that tile the full aperture.
        K = M - L + 1

        # Accumulate the forward sub-array covariance estimates.
        # Sub-array i spans virtual elements i through i+L-1, corresponding to the L x L
        # principal sub-matrix R[i:i+L, i:i+L] of the full covariance matrix.
        Rf = np.zeros((L, L), dtype=complex)
        for i in range(K):
            Rf += R[i:i + L, i:i + L]

        # Normalize by the number of sub-arrays to form the averaged forward estimate.
        Rf /= K

        # Construct the L x L exchange (reversal) matrix J.
        # J has ones on the anti-diagonal and zeros elsewhere. Left-multiplying by J reverses
        # the row order of a matrix; right-multiplying reverses the column order. Together,
        # J * A * J spatially reverses the array index ordering of A.
        J = np.fliplr(np.eye(L))  # exchange matrix

        # Derive the backward covariance estimate by conjugating and spatially reversing R_f.
        # This geometric reflection means R_b captures the same incoming signals from the
        # opposite array direction, introducing an independent rank contribution that further
        # decorrelates coherent sources.
        Rb = J @ Rf.conj() @ J

        # Return the Forward-Backward average. The factor of 0.5 normalizes the combination
        # so that the result remains a valid covariance estimate.
        return 0.5 * (Rf + Rb)

    @staticmethod
    def music_algorithm(R, M, num_sources, scan_angles):
        """
        Computes the MUSIC (MUltiple SIgnal Classification) pseudo-spectrum for a given
        spatial covariance matrix and a set of candidate scan angles.

        MUSIC exploits the eigenstructure of the spatial covariance matrix. For D impinging
        narrowband sources and M array elements, the M x M covariance matrix possesses D
        large eigenvalues associated with the signal subspace and M-D small eigenvalues
        (nominally equal to the noise variance) associated with the orthogonal noise
        subspace. The steering vectors of the true sources lie within the signal subspace
        and are therefore orthogonal to every column of the noise subspace basis.

        The MUSIC pseudo-spectrum is defined as:
            P_MUSIC(theta) = 1 / || E_n^H * a(theta) ||^2
        where a(theta) is the steering vector for angle theta and E_n is the noise subspace
        basis matrix. P_MUSIC is theoretically infinite at the true angles of arrival (where
        the denominator approaches zero) and small elsewhere, producing sharp spectral peaks.

        Parameters
        ----------
        R          : ndarray, shape (M, M)   Spatial covariance matrix (may be smoothed).
        M          : int                      Number of array elements represented in R.
        num_sources: int                      Estimated number of signal sources (D < M).
        scan_angles: ndarray, shape (N,)     Candidate scan angles in degrees.

        Returns
        -------
        P_music : ndarray, shape (N,)
            MUSIC pseudo-spectrum value at each candidate scan angle. Peaks identify the
            estimated directions of arrival.
        """
        # Decompose the covariance matrix via Singular Value Decomposition.
        # For a Hermitian positive-semidefinite covariance matrix, SVD is equivalent to an
        # eigendecomposition: U contains left singular vectors, Lambda contains singular
        # values ordered from largest to smallest, and Vh contains the corresponding right
        # singular vectors (conjugate transposes of the eigenvectors).
        U, Lambda, Vh = svd(R)

        # Extract the noise subspace basis: rows of Vh from index num_sources onward
        # correspond to the M-D smallest singular values (noise eigenvalues). Taking the
        # conjugate transpose yields an M x (M-D) matrix whose columns span the noise subspace.
        noise_subspace = Vh[num_sources:].conj().T  # shape: (M, M - num_sources)

        # Convert the candidate scan angles from degrees to radians for trigonometric computation.
        angles_rad = np.radians(scan_angles)

        # Construct the array steering matrix A of shape (M, N_angles).
        # For a uniform linear array with half-wavelength element spacing, the phase advance
        # from element 0 to element d for a source at angle theta is:
        #     phi_d(theta) = pi * d * sin(theta)
        # The factor pi = 2*pi * (d * lambda/2) / lambda accounts for the half-wavelength spacing.
        # np.outer generates all (element index, angle) combinations in a single vectorised operation.
        steering_matrix = np.exp(1j * np.pi * np.outer(np.arange(M), np.sin(angles_rad)))

        # Evaluate the MUSIC pseudo-spectrum for every candidate angle.
        # steering_matrix.conj().T @ noise_subspace projects each steering vector onto the
        # noise subspace. The squared norm of each row (np.linalg.norm with axis=1) measures
        # the energy in that projection. At a true source angle this norm approaches zero,
        # causing P_music to peak sharply.
        P_music = 1 / (np.linalg.norm(steering_matrix.conj().T @ noise_subspace, axis=1) ** 2)

        return P_music

    def visualization_azimuth(self, data: np.ndarray, detections_range_index: np.ndarray):
        """
        Plots the MUSIC azimuth pseudo-spectrum for each range bin that contains a detection.

        This method is intended for diagnostic and validation use. It iterates over all
        range bins in the detection index and invokes the azimuth spectrum plotter for every
        bin marked as containing a detection. The resulting plots allow the operator to
        visually confirm that MUSIC peaks appear at physically plausible azimuth angles for
        each detected target range and to identify any spurious or missed detections.

        Parameters
        ----------
        data                  : ndarray, shape (n_range, n_az)
            MUSIC pseudo-spectrum values indexed by [range bin, azimuth angle index].
        detections_range_index: ndarray, shape (n_range,), dtype bool
            Boolean mask indicating which range bins contain at least one CFAR detection.
        """
        # Iterate over every range bin index and check whether a detection was declared.
        for i in range(len(detections_range_index)):
            if detections_range_index[i]:
                # Convert the range bin index to a physical distance in metres and
                # invoke the azimuth spectrum plotter for this range gate.
                plot_azimuth_spectrum(
                    data[i],                               # pseudo-spectrum for this range bin
                    range_m=i * RANGE_RESOLUTION,          # physical range in metres
                    azimuth_fov_degrees=self.azimuth_fov_degrees,
                )

    @staticmethod
    def cfar_detection(data: np.ndarray, guard_doppler, guard_range,
                        train_doppler, train_range, k_rank, pfa, cfar_mode: str = "OS"):
        """
        Applies 2-D CFAR (Constant False Alarm Rate) detection to the range-Doppler power
        map and returns a boolean detection mask together with the per-cell threshold map.

        CFAR is an adaptive thresholding algorithm that maintains an approximately constant
        probability of false alarm regardless of the local noise and clutter level. Rather
        than comparing each cell under test (CUT) to a fixed global threshold, CFAR estimates
        the local noise power from a surrounding ring of training cells and multiplies that
        estimate by a scaling factor alpha (derived from the target pfa) to form the adaptive
        detection threshold. Guard cells immediately surrounding the CUT are excluded from
        the training window to prevent target energy from leaking into the noise estimate.

        Two CFAR variants are supported:

        OS-CFAR (Ordered-Statistics CFAR):
            Sorts the N training cell values in ascending order and selects the k-th value
            as the noise power estimate. The rank-order statistic is more robust than the mean
            at clutter boundaries and near the zero-Doppler ridge, where a minority of training
            cells may carry elevated clutter power. As long as fewer than N-k cells are
            contaminated by clutter, the rank-k estimate reflects the true noise floor.

        CA-CFAR (Cell-Averaging CFAR):
            Uses the arithmetic mean of all N training cell values as the noise power estimate.
            Optimal under homogeneous, stationary Gaussian noise, but degrades at clutter
            boundaries where training cells straddle regions of markedly different power levels,
            biasing the noise estimate and degrading detection performance.

        Parameters
        ----------
        data          : ndarray, shape (frames, doppler, range)
            Range-Doppler integrated power map.
        guard_doppler : int
            Number of guard cells excluded on each side of the CUT along the Doppler axis.
        guard_range   : int
            Number of guard cells excluded on each side of the CUT along the range axis.
        train_doppler : int
            Number of training cells on each side of the guard region along the Doppler axis.
        train_range   : int
            Number of training cells on each side of the guard region along the range axis.
        k_rank        : int or None
            Rank index for OS-CFAR (0 = smallest training cell value; N-1 = largest).
            Ignored when cfar_mode is "CA".
        pfa           : float
            Target probability of false alarm; controls the threshold multiplier alpha.
        cfar_mode     : str, default "OS"
            Detection variant: "OS" for Ordered-Statistics CFAR or "CA" for Cell-Averaging CFAR.

        Returns
        -------
        detection_mask : ndarray, shape (frames, doppler, range), dtype bool
            True at every cell whose power exceeds the adaptive detection threshold.
        threshold_map  : ndarray, shape (frames, doppler, range)
            Absolute detection threshold value computed for each cell.
        """
        # Dispatch to the appropriate CFAR implementation based on the requested mode.
        if cfar_mode == "OS":
            return os_cfar_on_cube(data, guard_doppler, guard_range, train_doppler, train_range, k_rank, pfa)
        elif cfar_mode == "CA":
            return ca_cfar_on_cube(data, guard_doppler, guard_range, train_doppler, train_range, pfa)
        else:
            raise ValueError("Invalid cfar mode. Expected either \"OS\" or \"CA\".")

    def point_cloud(self, data: List[np.ndarray]):
        """
        Converts per-frame detections expressed as (range index, azimuth index) pairs into
        Cartesian (x, y) coordinates in the radar's local coordinate frame.

        The coordinate convention used is:
            x = r * sin(az)   -> lateral (cross-range) axis, positive to the right
            y = r * cos(az)   -> longitudinal (down-range) axis, positive away from the radar

        Parameters
        ----------
        data : list of ndarray, each of shape (N, 2)
            Per-frame detection arrays. Each row contains [range_index, azimuth_index].

        Returns
        -------
        list of ndarray, each of shape (N, 2)
            Per-frame Cartesian point clouds with columns [x_m, y_m].
        """
        def to_xy(detections: np.ndarray):
            """Converts a single frame's detections from index space to Cartesian metres."""
            # Extract integer range and azimuth bin indices from the detection array columns.
            range_idx = detections[:, 0].astype(int)
            az_idx = detections[:, 1].astype(int)

            # Reconstruct the azimuth angle candidate axis used during MUSIC scanning.
            # The axis spans +-azimuth_fov_degrees at a uniform 0.1-degree angular step.
            az_candidates = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)

            # Convert range bin indices to physical range values in metres.
            r = range_idx * RANGE_RESOLUTION

            # Look up the physical azimuth angle (in radians) for each detection using its
            # azimuth bin index into the candidate angle axis.
            az = np.deg2rad(az_candidates[az_idx])

            # Project polar coordinates (r, az) into the Cartesian (x, y) plane.
            x = r * np.sin(az)   # lateral displacement in metres
            y = r * np.cos(az)   # longitudinal displacement in metres

            # Stack the x and y column vectors into a single (N, 2) coordinate array.
            return np.stack([x, y], axis=1)

        # Process each frame independently and collect the resulting Cartesian point clouds.
        frame_xy = []
        for frame_detections in data:
            xy = to_xy(frame_detections)
            frame_xy.append(xy)
        return frame_xy

    def steering_vector(self, azimuth):
        """
        Computes the array steering (manifold) vector for the virtual ULA at one or more
        azimuth angles.

        For a uniform linear array with half-wavelength element spacing, the phase advance
        from element 0 to element d for a source at azimuth angle theta is:
            phi_d(theta) = 2*pi * (d * lambda/2) * sin(theta) / lambda = pi * d * sin(theta)
        The steering vector element for element d is therefore:
            a_d(theta) = exp(j * pi * d * sin(theta))

        Since all virtual array elements are co-linear along the azimuth axis (the
        y-coordinate in self.pos is zero for every element), only the x-component of the
        spatial frequency kx = pi * sin(theta) contributes to the phase, and the element
        position simplifies to the x-index stored in self.pos[:, 0].

        Parameters
        ----------
        azimuth : float or array-like
            One or more azimuth angles in degrees.

        Returns
        -------
        ndarray, shape (n_rx, n_az)
            Complex steering matrix. Each column is the unit steering vector for one
            candidate azimuth angle.
        """
        # Convert azimuth angle(s) from degrees to radians.
        # atleast_1d ensures that a scalar input is promoted to a 1-D array so that
        # subsequent broadcasting operations are unambiguous.
        az = np.deg2rad(np.atleast_1d(azimuth))

        # Compute the spatial frequency kx = pi * sin(theta) for each candidate angle.
        # All virtual elements lie on the azimuth row (pos[:, 1] == 0), so only the
        # x-component of the wave-vector matters; the y-component contributes zero phase.
        kx = np.pi * np.sin(az)  # shape: (n_az,)

        # Compute the per-element, per-angle phase matrix via an outer product.
        # pos[:, 0] contains the x-coordinate (in half-wavelength units) of each element.
        # The resulting matrix has shape (n_rx, n_az), where entry [d, i] gives the
        # phase shift at element d for scan angle i.
        phase = self.pos[:, 0][:, None] * kx[None, :]  # shape: (n_rx, n_az)

        # Exponentiate the phase to produce the complex steering matrix.
        return np.exp(1j * phase)  # shape: (n_rx, n_az)

    def music_per_cell(self, snapshot: np.ndarray, scan_angles, L: int = 4, K: int = 1):
        """
        Applies the MUSIC algorithm to a single range-Doppler cell snapshot to estimate
        the azimuth angle(s) of the impinging signal(s).

        A single snapshot (one complex-valued vector of length M = n_rx) yields only a
        rank-1 outer-product covariance matrix when computed directly. Eigendecomposition
        of a rank-1 matrix does not provide a meaningful noise subspace (all but one
        eigenvalue are nominally zero), making direct MUSIC application degenerate.

        Forward-backward spatial smoothing with sub-array length L converts this rank-1
        matrix into a well-conditioned L x L estimate from which M-K noise-subspace
        dimensions can be extracted. MUSIC then resolves up to K sources within the single
        snapshot, at the cost of an effective aperture reduction from M to L elements.

        Parameters
        ----------
        snapshot    : ndarray, shape (n_rx,)
            Complex baseband signal vector from the virtual array for one (range, Doppler) cell.
        scan_angles : ndarray, shape (n_az,)
            Candidate azimuth angles in degrees at which to evaluate the pseudo-spectrum.
        L           : int, default 4
            Sub-array length for spatial smoothing. Must satisfy K < L <= n_rx.
        K           : int, default 1
            Estimated number of signal sources present in this (range, Doppler) cell.

        Returns
        -------
        P_music     : ndarray, shape (n_az,)
            MUSIC pseudo-spectrum evaluated at each candidate scan angle.
        scan_angles : ndarray, shape (n_az,)
            The same scan angle axis passed in, returned for downstream convenience.
        """
        # Form the rank-1 spatial covariance matrix from the single snapshot vector.
        # The outer product x * x^H produces an M x M Hermitian positive-semidefinite matrix
        # representing the instantaneous spatial correlation of the received signal.
        R = np.outer(snapshot, snapshot.conj())  # rank-1, shape: (n_rx, n_rx)

        # Apply forward-backward spatial smoothing to convert the rank-1 matrix into a
        # full-rank L x L covariance estimate that supports the eigendecomposition required
        # by MUSIC. The effective aperture is reduced from n_rx to L elements.
        R_smooth = self.fb_spatial_smoothing(R, L)  # shape: (L, L)

        # Run the MUSIC pseudo-spectrum computation on the smoothed covariance matrix.
        P_music = self.music_algorithm(R_smooth, L, K, scan_angles)

        return P_music, scan_angles

    def compute_music_heatmap(self, rd_cube: np.ndarray, frame_idx: int = 0, num_targets: int = 1):
        """
        Computes a 2-D range-azimuth MUSIC pseudo-spectrum heatmap for visualization and
        algorithm validation.

        For each range bin, the Doppler bin exhibiting the highest integrated power across
        all virtual receivers is selected as the representative snapshot for MUSIC processing.
        This strategy ensures that angle estimation is performed at the best available SNR
        operating point for each range gate. The resulting heatmap shows the distribution of
        target energy across the range-azimuth plane and can be used to verify that detected
        targets appear at physically expected angular positions and to observe the spatial
        smoothing effect.

        Input : rd_cube [frames, doppler, rx_virtual, range]
        Output: (heatmap [n_range, n_az], scan_angles [n_az])
        """
        # Select the target frame from the cube and unpack its dimensional extents.
        frame = rd_cube[frame_idx]          # shape: (doppler, rx_virtual, range)
        n_doppler, n_rx, n_range = frame.shape

        # Compute the per-Doppler-bin power integrated across all virtual receivers.
        # Summing the squared magnitudes over the receiver axis collapses the spatial
        # dimension, producing a (doppler, range) scalar power map used solely for
        # identifying the highest-SNR Doppler snapshot at each range gate.
        doppler_power = (np.abs(frame) ** 2).sum(axis=1)   # shape: (doppler, range)

        # For each range bin, find the Doppler bin index with the maximum integrated power.
        # This is the Doppler snapshot that contains the target's energy at the best SNR.
        peak_dop = np.argmax(doppler_power, axis=0)         # shape: (range,)

        # Define the azimuth scan grid: uniformly spaced angles within the FOV at 0.1-degree steps.
        scan_angles = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)

        # Allocate the output heatmap array; each row corresponds to one range bin.
        heatmap = np.zeros((n_range, len(scan_angles)))

        # Iterate over every range bin and compute its MUSIC azimuth pseudo-spectrum.
        for r in range(n_range):
            # Extract the spatial snapshot at the peak-power Doppler bin for this range gate.
            # x is a complex vector of length n_rx representing the signal seen by each
            # virtual element at the chosen Doppler and range coordinates.
            x = frame[peak_dop[r], :, r]   # shape: (rx_virtual,)

            # Apply MUSIC to the snapshot; K is set to the estimated number of targets.
            P, _ = self.music_per_cell(x, scan_angles, K=num_targets)

            # Store the resulting pseudo-spectrum as one row of the heatmap.
            heatmap[r] = P

        return heatmap, scan_angles

    def point_cloud_from_rd_azimuth(self, rd_cube: np.ndarray,
                            rd_det: np.ndarray, K: int = 1):
        """
        Runs the MUSIC angle estimator on every CFAR-detected range-Doppler cell and
        assembles the resulting measurements into a per-frame point cloud.

        For each cell flagged by the CFAR detector, the complex virtual-array snapshot is
        extracted and passed to MUSIC. The pseudo-spectrum is converted to decibels and a
        prominence-based peak-finding algorithm identifies angle candidates corresponding to
        genuine sources rather than noise-floor ripple. Up to K strongest peaks are retained
        per cell, each contributing one point to the cloud.

        Each output point is characterized by four attributes:
            range_m    : physical slant range of the detection in metres.
            velocity   : signed radial velocity in m/s (positive = receding from radar,
                         negative = approaching the radar).
            azimuth_deg: estimated angle of arrival in degrees.
            power_dBFS : total received signal power of the virtual-array snapshot in dBFS.

        Inputs
        ------
        rd_cube  : [frames, doppler, rx_virtual, range]   (phase-corrected)
        rd_det   : [frames, doppler, range]               bool, from CFAR

        Returns
        -------
        list of length n_frames; each entry is (n_pts, 4) with columns
        (range_m, velocity_m_s, azimuth_deg, power_dBFS).
        """
        # Unpack the number of frames and Doppler bins from the cube shape.
        n_frames, n_doppler, _, _ = rd_cube.shape

        # Total inter-TX delay used to convert Doppler bin indices to physical velocity.
        T_loop = TX_CHANNELS * CHIRP_CYCLE_TIME

        # Construct the signed (centred) Doppler bin index array matching the fftshift
        # convention. Subtracting n_doppler // 2 maps bin 0 to -N/2 and bin N-1 to N/2-1.
        doppler_idx = np.arange(n_doppler) - n_doppler // 2

        # Convert Doppler bin indices to signed radial velocities in m/s.
        # The radar velocity equation is: v = (bin_index * lambda) / (2 * N_doppler * T_loop).
        # A positive velocity indicates a target receding from the radar;
        # a negative velocity indicates a target approaching the radar.
        v_axis = doppler_idx * WAVELENGTH / (2 * n_doppler * T_loop)

        # Define the azimuth scan grid used by MUSIC (same grid as the heatmap computation).
        scan_angles = np.arange(-self.azimuth_fov_degrees, self.azimuth_fov_degrees, 0.1)

        out = []
        for f in range(n_frames):
            # Extract the binary CFAR detection mask for this frame.
            det = rd_det[f]  # shape: (doppler, range), dtype bool

            # Retrieve the (Doppler, range) bin index pairs of all declared detections.
            d_idx, r_idx = np.where(det)

            pts = []
            for d, r in zip(d_idx, r_idx):
                # Extract the complex virtual-array snapshot for this (Doppler, range) cell.
                # x captures the spatial phase pattern across all virtual receivers,
                # encoding the angle-of-arrival information required by MUSIC.
                x = rd_cube[f, d, :, r]  # shape: (rx_virtual,)

                # Run MUSIC on the snapshot to obtain the azimuth pseudo-spectrum.
                P, scan = self.music_per_cell(x, scan_angles, K=K)

                # Convert the pseudo-spectrum to decibels for prominence-based peak detection.
                # The additive offset 1e-12 prevents numerical overflow from log(0) at
                # near-zero spectrum values.
                P_db = 10 * np.log10(P + 1e-12)

                # Identify local maxima in the dB pseudo-spectrum.
                # A minimum prominence of 6 dB is required to distinguish a genuine source
                # peak from noise-floor ripple or MUSIC sidelobe artifacts.
                peak_idx, _ = find_peaks(P_db, prominence=6)

                # Rank the detected peaks by their linear-scale pseudo-spectrum amplitude
                # (descending order) and retain at most K peaks, one per expected source.
                order = np.argsort(P[peak_idx])[::-1][:K]

                # Compute the total received power of this snapshot in dBFS.
                # Summing the squared magnitudes across all virtual receivers gives the
                # total incoherent power from all spatial directions in this cell.
                power_db = 10 * np.log10(np.sum(np.abs(x) ** 2) + 1e-12)

                # Append one 4-tuple per retained MUSIC peak to the frame's detection list.
                for pk in peak_idx[order]:
                    pts.append((r * RANGE_RESOLUTION, v_axis[d], scan[pk], power_db))

            # Convert the list of 4-tuples to a NumPy array.
            # If no detections were found in this frame, return an empty (0, 4) array
            # to maintain a consistent structure across all frames.
            out.append(np.array(pts) if pts else np.empty((0, 4)))

        return out


if __name__ == '__main__':
    adc_data = parse_mmwave_studio_adc(RAW_ADC_PATH, ADC_SAMPLES, N_CHIRPS_LOOPS, RX_CHANNELS, 2, N_FRAMES)

    p = PointCloud1DMUSIC()

    p.run(adc_data)
