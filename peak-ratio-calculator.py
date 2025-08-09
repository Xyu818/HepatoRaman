#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian(x, amp, cen, sigma):
    """Gaussian peak function"""
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

def main():

    # Read spectrum: two columns -> wavenumber, intensity
    df = pd.read_csv(
        'spectrum.txt',
        sep=r'\s+',
        comment='#',
        names=['wavenumber', 'intensity']
    )

    # Smooth intensity with Savitzky–Golay filter
    df['smoothed'] = savgol_filter(df['intensity'], window_length=11, polyorder=3)

    # Target peak positions (cm^-1)
    targets = [500, 650, 720, 890, 1035, 1100, 1185]
    peak_info = {}

    # Find and (optionally) fit a Gaussian near each target
    for t in targets:
        mask = (df['wavenumber'] >= t - 10) & (df['wavenumber'] <= t + 10)
        wn_win = df['wavenumber'][mask].values
        int_win = df['smoothed'][mask].values

        # local peak detection
        peaks, _ = find_peaks(int_win)
        if peaks.size == 0:
            print(f"Warning: no peak detected near {t} cm⁻¹")
            continue

        # initial guess from the highest local peak
        idx0 = peaks[np.argmax(int_win[peaks])]
        init_amp = int_win[idx0]
        init_cen = wn_win[idx0]

        # Gaussian fit; if it fails, fall back to the raw peak
        try:
            popt, _ = curve_fit(
                gaussian,
                wn_win,
                int_win,
                p0=[init_amp, init_cen, 5]
            )
            amp, cen, sigma = popt
        except RuntimeError:
            amp, cen = init_amp, init_cen

        peak_info[t] = {'center': cen, 'amplitude': amp}

    # Compute ratios to the 500 cm^-1 peak, if available
    if 500 in peak_info:
        ref_amp = peak_info[500]['amplitude']
        ratios = {
            f"500/{t}": peak_info[t]['amplitude'] / ref_amp
            for t in peak_info if t != 500
        }
    else:
        print("Error: 500 cm⁻¹ peak not found; cannot compute ratios.")
        ratios = {}

    # Summarize results
    results = pd.DataFrame([
        {
            'target_cm-1': t,
            'peak_center': info['center'],
            'peak_amplitude': info['amplitude'],
            'ratio_to_500': (info['amplitude'] / ref_amp) if 500 in peak_info else np.nan
        }
        for t, info in peak_info.items()
    ])
    print("\n=== Peak detection & fitting results ===")
    print(results.to_string(index=False))

    # Plot smoothed spectrum with detected peak centers
    plt.figure(figsize=(8, 4))
    plt.plot(df['wavenumber'], df['smoothed'], label='Smoothed spectrum')
    for t, info in peak_info.items():
        plt.axvline(info['center'], linestyle='--', alpha=0.7,
                    label=f"{t} cm⁻¹ → {info['center']:.1f}")
    plt.gca().invert_xaxis()
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Raman spectrum with detected peaks')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
