"""
================================================================================
Automated 327 MHz RFI Mitigation and Signal Reconstruction Pipeline
Designed for Ooty Radio Telescope (ORT) Baseband Data (Vela Pulsar / PSR J0835-4510)
--------------------------------------------------------------------------------
Author: VSRP Research Candidate
Target File: C:\\Users\\dhars\\Desktop\\IUCAA\\Iucaa Project (TRAINING WORKING GROUP)\\pulsar_data1.txt
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PIPELINE CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
DATA_FILE_PATH = r"C:\Users\dhars\Desktop\IUCAA\Iucaa Project (TRAINING WORKING GROUP)\pulsar_data1.txt"

# Physical and Instrumentation Parameters
DT_SAMPLE = 30.30e-9       # Sampling time: ~30.30 ns (Nyquist sampling for 16.5 MHz bandwidth)
N_FFT = 512                # FFT block size -> 256 frequency channels
TIME_AVG = 32              # Number of consecutive FFT blocks to co-add (~0.5 ms time resolution)
F_CENTER = 326.5           # ORT observing center frequency (MHz)
BANDWIDTH = 16.5           # Baseband bandwidth (MHz) -> 0 to 16.5 MHz
VELA_PERIOD = 0.0893       # Rotational period of Vela pulsar (seconds) ~89.3 ms
VELA_DM = 67.99            # Dispersion Measure (pc cm^-3)
K_DISPERSION = 4.148808e3  # Dispersion constant in s * MHz^2 / (pc cm^-3)

# Set to True because ORT receiver down-conversion typically inverts the spectrum
FLIP_SPECTRUM = True       

# -----------------------------------------------------------------------------
# STEP 1: LOAD RAW VOLTAGE TIME-SERIES
# -----------------------------------------------------------------------------
def load_raw_voltages(file_path):
    """
    Reads 2-column ASCII text file containing uncalibrated integer voltages.
    Column 0: North Aperture Voltages
    Column 1: South Aperture Voltages
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}\nPlease verify the Windows path.")

    print(f"[Step 1] Loading raw voltages from: {file_path}")
    raw_data = np.loadtxt(file_path)
    v_north = raw_data[:, 0]
    v_south = raw_data[:, 1]
    
    print(f" -> Successfully loaded {len(raw_data):,} voltage samples per channel.")
    print(f" -> North Channel: Mean = {np.mean(v_north):.4f}, Std Dev = {np.std(v_north):.4f}")
    print(f" -> South Channel: Mean = {np.mean(v_south):.4f}, Std Dev = {np.std(v_south):.4f}")
    return v_north, v_south

# -----------------------------------------------------------------------------
# STEP 2: DYNAMIC SPECTRUM GENERATION (FFT + CO-ADDING + BANDPASS NORM)
# -----------------------------------------------------------------------------
def compute_dynamic_spectrum(v_north, v_south, n_fft=N_FFT, dt_sample=DT_SAMPLE, time_avg=TIME_AVG, flip=FLIP_SPECTRUM):
    """
    Transforms 1D voltage time-series from North & South apertures into a combined,
    time-averaged, and bandpass-normalized 2D Dynamic Spectrum.
    """
    print("\n[Step 2] Computing & Combining North + South Dynamic Spectra...")
    num_blocks = len(v_north) // n_fft
    num_chans = n_fft // 2

    # Process North Channel
    vn_blocks = v_north[:num_blocks * n_fft].reshape((num_blocks, n_fft))
    spec_north = np.abs(np.fft.rfft(vn_blocks, axis=1)[:, :num_chans])**2

    # Process South Channel
    vs_blocks = v_south[:num_blocks * n_fft].reshape((num_blocks, n_fft))
    spec_south = np.abs(np.fft.rfft(vs_blocks, axis=1)[:, :num_chans])**2

    # Combine power from both apertures (incoherent sum boosts SNR by ~sqrt(2))
    raw_spec = spec_north + spec_south

    # Correct for spectral inversion from down-conversion
    if flip:
        raw_spec = raw_spec[:, ::-1]

    # Co-add consecutive spectra along time axis to improve SNR per bin
    n_avg_bins = num_blocks // time_avg
    dynamic_spectrum = raw_spec[:n_avg_bins * time_avg].reshape(n_avg_bins, time_avg, num_chans).mean(axis=1)

    # Bandpass Normalization: divide each frequency channel by its median power
    bandpass = np.median(dynamic_spectrum, axis=0)
    bandpass[bandpass == 0] = 1.0  # Prevent division by zero
    dynamic_spectrum = dynamic_spectrum / bandpass

    dt_bin = n_fft * time_avg * dt_sample
    df_chan = BANDWIDTH / num_chans

    print(f" -> Dynamic Spectrum Matrix Shape: {dynamic_spectrum.shape} (Time Bins x Freq Channels)")
    print(f" -> Time Resolution per Bin: {dt_bin * 1e3:.3f} milliseconds")
    print(f" -> Channel Width: {df_chan * 1e3:.2f} kHz")
    return dynamic_spectrum, dt_bin, df_chan

# -----------------------------------------------------------------------------
# STEP 3: AUTOMATED RFI MASKING ENGINE (ENHANCED MAD THRESHOLDING)
# -----------------------------------------------------------------------------
def apply_rfi_masking(dyn_spec, k_sigma=2.5):
    """
    Automated RFI Cleaner using Median Absolute Deviation (MAD) on bandpass-normalized data.
    Catches both persistent narrowband noise and intermittent RFI spikes.
    """
    print("\n[Step 3] Running Automated RFI Cleaning Engine...")
    cleaned_spec = dyn_spec.copy()
    
    # 1. Narrowband RFI Detection (combining channel std dev + peak power)
    channel_metric = np.std(cleaned_spec, axis=0) + 0.2 * np.max(cleaned_spec, axis=0)
    med_metric = np.median(channel_metric)
    mad_chan = np.median(np.abs(channel_metric - med_metric))
    thresh_chan = med_metric + k_sigma * (1.4826 * mad_chan)
    
    bad_channels = np.where(channel_metric > thresh_chan)[0]
    cleaned_spec[:, bad_channels] = 1.0  # Replace with normalized baseline (1.0)
    
    # 2. Broadband Burst RFI Detection across Time Slices
    time_medians = np.median(cleaned_spec, axis=1)
    global_time_med = np.median(time_medians)
    mad_time = np.median(np.abs(time_medians - global_time_med))
    thresh_time = global_time_med + k_sigma * (1.4826 * mad_time)
    
    bad_time_bins = np.where(time_medians > thresh_time)[0]
    cleaned_spec[bad_time_bins, :] = global_time_med
    
    print(f" -> Flagged & Masked {len(bad_channels)} Narrowband RFI Channels.")
    print(f" -> Flagged & Masked {len(bad_time_bins)} Broadband RFI Time Slices.")
    return cleaned_spec, bad_channels, bad_time_bins

# -----------------------------------------------------------------------------
# STEP 4: INTERSTELLAR DE-DISPERSION ALIGNMENT
# -----------------------------------------------------------------------------
def dedisperse_spectrum(dyn_spec, dt_bin, dm=VELA_DM, f_center=F_CENTER, bw=BANDWIDTH):
    """
    Corrects for frequency-dependent interstellar dispersion delay:
    dt = 4.1488e3 * DM * (f1^-2 - f2^-2)
    """
    print(f"\n[Step 4] Applying Incoherent De-dispersion for DM = {dm} pc cm^-3...")
    num_time_bins, num_chans = dyn_spec.shape
    f_low = f_center - (bw / 2.0)
    f_high = f_center + (bw / 2.0)
    
    # Calculate exact RF frequency for each channel
    freqs = np.linspace(f_low, f_high, num_chans)
    
    # Delay relative to highest frequency (f_high)
    delays_sec = K_DISPERSION * dm * ((1.0 / freqs**2) - (1.0 / f_high**2))
    shifts_bins = np.round(delays_sec / dt_bin).astype(int)
    
    dedisp_spec = np.zeros_like(dyn_spec)
    for ch in range(num_chans):
        shift = shifts_bins[ch]
        dedisp_spec[:, ch] = np.roll(dyn_spec[:, ch], -shift)
        
    print(f" -> Maximum dispersion delay shift: {np.max(shifts_bins)} time bins ({np.max(delays_sec)*1e3:.2f} ms).")
    return dedisp_spec, shifts_bins

# -----------------------------------------------------------------------------
# STEP 5: PHASE FOLDING & PULSE PROFILE EXTRACTION
# -----------------------------------------------------------------------------
def fold_pulse_profile(dyn_spec, dt_bin, period=VELA_PERIOD, n_phase_bins=64):
    """
    Folds the integrated time-series over the pulsar rotational period.
    """
    time_series = np.mean(dyn_spec, axis=1)  # Average across frequency channels
    num_time_bins = len(time_series)
    
    times = np.arange(num_time_bins) * dt_bin
    phases = (times % period) / period
    
    phase_bins = np.floor(phases * n_phase_bins).astype(int)
    
    folded_profile = np.zeros(n_phase_bins)
    counts = np.zeros(n_phase_bins)
    
    for i in range(num_time_bins):
        bin_idx = phase_bins[i]
        folded_profile[bin_idx] += time_series[i]
        counts[bin_idx] += 1
        
    counts[counts == 0] = 1
    folded_profile /= counts
    
    # Normalize profile baseline to zero and scale by off-pulse std dev (SNR)
    folded_profile = (folded_profile - np.median(folded_profile)) / np.std(folded_profile)
    return folded_profile

# -----------------------------------------------------------------------------
# DIAGNOSTIC PLOTTING & PIPELINE EXECUTION
# -----------------------------------------------------------------------------
def run_pipeline():
    # 1. Load Data
    v_north, v_south = load_raw_voltages(DATA_FILE_PATH)
    
    # 2. Compute Combined & Normalized Dynamic Spectrum
    raw_dyn_spec, dt_bin, df_chan = compute_dynamic_spectrum(v_north, v_south)
    
    # 3. Apply Enhanced RFI Masking
    cleaned_dyn_spec, bad_ch, bad_t = apply_rfi_masking(raw_dyn_spec, k_sigma=2.5)
    
    # 4. De-disperse Raw vs Cleaned Spectra
    raw_dedisp, _ = dedisperse_spectrum(raw_dyn_spec, dt_bin=dt_bin)
    cleaned_dedisp, _ = dedisperse_spectrum(cleaned_dyn_spec, dt_bin=dt_bin)
    
    # 5. Fold Profiles
    print("\n[Step 5] Folding Raw vs Cleaned Profiles...")
    n_phase_bins = 64
    raw_profile = fold_pulse_profile(raw_dedisp, dt_bin=dt_bin, n_phase_bins=n_phase_bins)
    cleaned_profile = fold_pulse_profile(cleaned_dedisp, dt_bin=dt_bin, n_phase_bins=n_phase_bins)
    
    # 6. Plot Comprehensive Diagnostic Overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Clip extreme outlier spikes for better visual contrast in dynamic spectra
    vmin, vmax = np.percentile(cleaned_dyn_spec, [2, 98])
    
    # Plot 1: Raw Dynamic Spectrum
    im0 = axes[0, 0].imshow(raw_dyn_spec.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("1. Bandpass-Normalized Dynamic Spectrum (Time vs Freq)")
    axes[0, 0].set_xlabel(f"Time Bins ({dt_bin*1e3:.2f} ms/bin)")
    axes[0, 0].set_ylabel("Frequency Channels (318.25 - 334.75 MHz)")
    plt.colorbar(im0, ax=axes[0, 0], label="Normalized Power")
    
    # Plot 2: RFI Masked Dynamic Spectrum
    im1 = axes[0, 1].imshow(cleaned_dyn_spec.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"2. Automated RFI Masked Spectrum ({len(bad_ch)} ch, {len(bad_t)} time bins masked)")
    axes[0, 1].set_xlabel(f"Time Bins ({dt_bin*1e3:.2f} ms/bin)")
    axes[0, 1].set_ylabel("Frequency Channels")
    plt.colorbar(im1, ax=axes[0, 1], label="Normalized Power")
    
    # Plot 3: De-dispersed Dynamic Spectrum
    im2 = axes[1, 0].imshow(cleaned_dedisp.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("3. De-dispersed Spectrum (DM = 67.99 pc cm^-3)")
    axes[1, 0].set_xlabel(f"Time Bins ({dt_bin*1e3:.2f} ms/bin)")
    axes[1, 0].set_ylabel("Frequency Channels")
    plt.colorbar(im2, ax=axes[1, 0], label="Normalized Power")
    
    # Plot 4: Folded Pulse Profile Comparison
    phase_axis = np.linspace(0, 1, n_phase_bins, endpoint=False)
    axes[1, 1].plot(phase_axis, raw_profile, color='red', alpha=0.6, label='Before RFI Masking')
    axes[1, 1].plot(phase_axis, cleaned_profile, color='blue', linewidth=2, label='Cleaned Pipeline Profile')
    axes[1, 1].set_title("4. Integrated Pulse Profile (Vela Pulsar)")
    axes[1, 1].set_xlabel("Rotational Phase (0.0 to 1.0)")
    axes[1, 1].set_ylabel("Signal-to-Noise Ratio (SNR)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_plot_path = "ort_pulsar_pipeline_results.png"
    plt.savefig(output_plot_path, dpi=300)
    print(f"\n[Success] Diagnostic figure saved to: {output_plot_path}")
    plt.show()

if __name__ == "__main__":
    run_pipeline()