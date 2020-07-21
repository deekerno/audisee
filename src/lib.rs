//! # audisee
//!
//! This library provides several methods for extracting features from audio
//! data. It provides a foundation upon which higher-level audio information
//! systems can be built.

#![warn(
    anonymous_parameters,
    missing_copy_implementations,
    missing_debug_implementations,
    missing_docs,
    nonstandard_style,
    rust_2018_idioms,
    single_use_lifetimes,
    trivial_casts,
    trivial_numeric_casts,
    unreachable_pub,
    unused_extern_crates,
    unused_qualifications,
    variant_size_differences
)]

pub mod spectral;
pub mod temporal;
pub mod utils;

use num_complex::Complex;
use rustfft::FFTplanner;

/// Returns the amplitude spectrum of a signal block.
///
/// In the amplitude spectrum, each index is a frequency bin; each bin is a
/// range of frequencies. Each element in the spectrum is the amplitude
/// corresponding to that frequency bin. All of the spectral features make
/// use of the frequencies calculated in the spectrum as well as their
/// respective magnitudes.
pub fn amp_spectrum(signal: &[f64]) -> Vec<f64> {
    let signal_length = signal.len();
    let fft = FFTplanner::new(false).plan_fft(signal_length);

    let mut complex_signal = signal
        .iter()
        .map(|x| Complex::new(*x, 0f64))
        .collect::<Vec<Complex<f64>>>();

    let mut spectrum = vec![Complex::new(0_f64, 0_f64); signal_length];

    fft.process(&mut complex_signal, &mut spectrum);

    spectrum
        .iter()
        .take(spectrum.len() / 2)
        .map(|bin| (bin.re.powi(2) + bin.im.powi(2)).sqrt())
        .collect::<Vec<f64>>()
}

/// Returns the power spectrum of a signal block.
///
/// This differs from an amplitude spectrum as it emphasizes the differences
/// between frequency bins.
pub fn power_spectrum(amp_spectrum: &[f64]) -> Vec<f64> {
    amp_spectrum.iter().map(|bin| bin.powi(2)).collect()
}

/// Returns the output of the fast Fourier transform using a signal as input.
///
/// The output of the FFT is a complex vector. It contains information of the
/// frequencies present in a signal, namely the magnitudes and phases of each
/// frequency. It is exposed here in order to enable operations that are not
/// included in the library at this time.
pub fn fft(signal: &[f64]) -> Vec<Complex<f64>> {
    let signal_length = signal.len();
    let fft = FFTplanner::new(false).plan_fft(signal_length);

    let mut complex_signal = signal
        .iter()
        .map(|x| Complex::new(*x, 0f64))
        .collect::<Vec<Complex<f64>>>();

    let mut spectrum = vec![Complex::new(0_f64, 0_f64); signal_length];

    fft.process(&mut complex_signal, &mut spectrum);

    spectrum
}
