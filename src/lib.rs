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

pub mod chroma;
pub mod spectral;
pub mod temporal;
pub mod utils;

use apodize;
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
    // Apply a Hanning window in order to smooth things out.
    let windowed_signal = apply_window(signal);

    let spectrum = fft(&windowed_signal[..]);

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

/// Returns the amplitude spectrum of an entire signal through time.
///
/// This function takes the input signal (an audio time-series) and pads it before generating
/// overlapping frames. Each overlapping frame is windowed using the Hann function and then
/// processed through FFT. An amplitude (magnitude) spectrum is created for each of the resultant
/// complex spectrums. The output is a vector of amplitude spectrums for each overlapping frame.
pub fn spectrogram(signal: &[f64]) -> Vec<Vec<f64>> {
    // Pad the signal before calculating the FFT.
    let padding_needed = signal.len() % 4;
    let padded_signal = if padding_needed != 0 {
        let mut signal_clone = signal.to_vec().clone();
        let mut padding = vec![0_f64; padding_needed];
        signal_clone.append(&mut padding);
        signal_clone
    } else {
        signal.to_vec().clone()
    };

    // Then, generate overlapping frames for the signal.
    let frame_size = padded_signal.len() / 4;
    let overlapping_frames = utils::OverlappingFrames::new(&padded_signal, frame_size, 0.75);

    // Re-use FFT planner as recommended in RustFFT docs.
    let mut planner = FFTplanner::new(false);

    // Reduce likelihood of decreased performance due to constant allocation
    // by pre-allocating a vector to a capacity equal to the amount of frames.
    let (_, upper_bound) = overlapping_frames.size_hint();
    let capacity = match upper_bound {
        Some(n) => n,
        None => frame_size * 2,
    };
    let mut amp_spec_frames = Vec::with_capacity(capacity);

    for frame in overlapping_frames {
        let fft = planner.plan_fft(frame.len());

        // Apply window to the padded signal to prevent jankiness.
        let windowed_frame = apply_window(&frame[..]);

        let mut complex_frame = windowed_frame
            .iter()
            .map(|x| Complex::new(*x, 0f64))
            .collect::<Vec<Complex<f64>>>();

        let mut spectrum = vec![Complex::new(0_f64, 0_f64); complex_frame.len()];
        fft.process(&mut complex_frame, &mut spectrum);

        // Create magnitude spectrum for the frame.
        let spec_frame = spectrum
            .iter()
            .take(spectrum.len() / 2)
            .map(|bin| (bin.re.powi(2) + bin.im.powi(2)).sqrt())
            .collect::<Vec<f64>>();

        amp_spec_frames.push(spec_frame);
    }

    amp_spec_frames
}

/// Returns a vector comprised of the calculated feature for every frame in a spectrogram.
///
/// The spectrogram is a time series of the input signal's frequencies and their respective
/// magnitudes. This method makes it possible to use the spectral feature extraction methods across
/// the entire time series.
pub fn generate_feature_time_series(
    f: fn(&[f64], Option<f64>) -> f64,
    spectrogram: &Vec<Vec<f64>>,
    sample_rate: Option<f64>,
) -> Vec<f64> {
    spectrogram
        .iter()
        .map(|frame| f(&frame[..], sample_rate))
        .collect::<Vec<f64>>()
}

/// Returns the output of the fast Fourier transform using a signal as input.
///
/// The output of the FFT is a complex vector. It contains information of the
/// frequencies present in a signal, namely the magnitudes and phases of each
/// frequency. It is exposed here in order to enable operations that are not
/// included in the library at this time.
fn fft(signal: &[f64]) -> Vec<Complex<f64>> {
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

fn apply_window(signal: &[f64]) -> Vec<f64> {
    let signal_length = signal.len();
    let window = apodize::hanning_iter(signal_length);

    let windowed_signal: Vec<f64> = signal
        .iter()
        .zip(window)
        .map(|(sample, window)| sample * window)
        .collect();

    windowed_signal
}
