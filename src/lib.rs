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
pub fn power_spectrum(signal: &[f64]) -> Vec<f64> {
    let amp_spectrum = amp_spectrum(signal);

    amp_spectrum.iter().map(|bin| bin.powi(2)).collect()
}

/// Calculates the energy of a signal.
///
/// This can be used to determine the loudness of a signal.
pub fn energy(signal: &[f64]) -> f64 {
    signal
        .iter()
        .fold(0_f64, |acc, &sample| acc + sample.abs().powi(2))
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

/// Calculates the root mean square of a signal.
///
/// This can be used to determine the loudness of a signal.
pub fn rms(signal: &[f64]) -> f64 {
    let sum = signal
        .iter()
        .fold(0_f64, |acc, &sample| acc + sample.powi(2));

    let mean = sum / signal.len() as f64;

    mean.sqrt()
}

/// Calculates the spectral centroid of a signal.
///
/// The spectral centroid represents the "center of gravity" for a spectrum.
/// It is often used to determine the timbre (perceived brightness) of a sound.
pub fn spectral_centroid(signal: &[f64]) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| acc + (idx as f64 * val));

    let denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    numerator / denominator
}

/// Calculates the spectral crest of a signal.
///
/// Spectral crest can be used to determine the peakiness of a spectrum. A
/// higher spectral crest indicates more tonality, while a lower spectral
/// crest denotes more noise.
pub fn spectral_crest(signal: &[f64]) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let numerator = amp_spectrum
        .iter()
        .fold(std::f64::NEG_INFINITY, |a, b| a.max(*b));

    let denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x) / amp_spectrum.len() as f64;

    numerator / denominator
}

/// Calculates the spectral decrease of a signal.
///
/// Spectral decrease represents the amount of decrease in a spectrum, while emphasizing the slopes
/// of lower frequencies. In tandem with other measures, it can be used for instrument detection.
pub fn spectral_decrease(signal: &[f64]) -> f64 {
    let mut amp_spectrum = amp_spectrum(signal);

    // Maybe use a drain here?
    let s_b1 = amp_spectrum.remove(0);

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + ((val - s_b1) / (idx as f64 - 1_f64))
        });

    let denominator: f64 = amp_spectrum.iter().sum();

    numerator / denominator
}

/// Calculates the spectral entropy of a signal.
///
/// Spectral entropy measures the peakiness of a spectrum. As entropy is a
/// measure of disorder, it can be used to differentiate between types of
/// sound that have different expectations of "order", e.g. speech vs.
/// music with multiple instruments.
pub fn spectral_entropy(signal: &[f64]) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let numerator = -amp_spectrum
        .iter()
        .fold(0_f64, |acc, x| acc + (x * x.log10()));

    let denominator = amp_spectrum.len() as f64;
    let denominator = denominator.log10();

    numerator / denominator
}

/// Calculates the spectral flatness of a signal.
///
/// Spectral flatness can be used to determine the peakiness of a spectrum. A
/// higher spectral flatness indicates more tonality, while a lower spectral
/// flatness denotes more noise.
pub fn spectral_flatness(signal: &[f64]) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let numerator = amp_spectrum
        .iter()
        .fold(0_f64, |acc, x| acc * x)
        .powf(1_f64 / amp_spectrum.len() as f64);

    let denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x) / amp_spectrum.len() as f64;

    numerator / denominator
}

/// Calculates the spectral flux of a signal.
///
/// Spectral flux is a measure of the variability of the spectrum over time.
pub fn spectral_flux(signal: &[f64]) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let diff_length = amp_spectrum.len() - 1;

    let mut diffs = Vec::new();

    for idx in 1..diff_length {
        let diff = amp_spectrum[idx] - amp_spectrum[idx - 1];
        diffs.push(diff);
    }

    let squared_sum = diffs.iter().fold(0_f64, |acc, x| acc + x.abs().powi(2));

    squared_sum.sqrt()
}

/// Calculates the spectral kurtosis of a signal.
///
/// Spectral kurtosis measures the flatness of a spectrum near its centroid
/// ("center of gravity"). It can also be used to measure the peakiness of
/// a spectrum as well.
pub fn spectral_kurtosis(signal: &[f64]) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let (mu_1, mu_2) = centroid_and_spread(&amp_spectrum);

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + ((idx as f64 - mu_1).powi(4) * val)
        });

    let denominator = mu_2.powi(4) * amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    numerator / denominator
}

/// Calculates the spectral rolloff point of a signal.
///
/// The spectral rolloff point is the frequency bin under which a given percentage of the total
/// energy in a spectrum exists. It can be used to distinguish unique types of audio in many
/// different situations.
pub fn spectral_rolloff(
    signal: &[f64],
    sampling_rate: Option<f64>,
    energy_threshold: Option<f64>,
) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let mut total_energy: f64 = amp_spectrum.iter().sum();

    let samp_rate = match sampling_rate {
        Some(sr) => sr,
        None => 44100_f64,
    };
    let bin_transform = (samp_rate * amp_spectrum.len() as f64) / 2_f64;

    let thresholded_total = match energy_threshold {
        Some(et) => et * total_energy,
        None => 0.85 * total_energy,
    };

    let mut idx = amp_spectrum.len() - 1;
    while total_energy > thresholded_total {
        total_energy -= amp_spectrum[idx];
        idx -= 1;
    }

    (idx + 1) as f64 * bin_transform
}

/// Calculates the spectral skewness of a signal.
///
/// Spectral skewness measures symmetry around the centroid.
pub fn spectral_skewness(signal: &[f64]) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let (mu_1, mu_2) = centroid_and_spread(&amp_spectrum);

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + ((idx as f64 - mu_1).powi(3) * val)
        });

    let denominator = mu_2.powi(3) * amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    numerator / denominator
}

/// Calculates the spectral slope of a signal.
///
/// Spectral energy measures the amount of decrease of the spectrum. It is most pronounced when the
/// energy in the lower formants is much greater than the energy of the higer formants.
pub fn spectral_slope(signal: &[f64], sampling_rate: Option<f64>) -> f64 {
    let amp_spectrum = amp_spectrum(signal);

    let samp_rate = match sampling_rate {
        Some(sr) => sr,
        None => 44100_f64,
    };
    let bin_transform = (samp_rate * amp_spectrum.len() as f64) / 2_f64;

    let amp_sum: f64 = amp_spectrum.iter().sum();
    let amp_mean: f64 = amp_sum / amp_spectrum.len() as f64;
    let freq_mean: f64 = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, _)| acc + (idx as f64 * bin_transform))
        / amp_spectrum.len() as f64;

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + (((idx as f64 * bin_transform) - freq_mean) * (val - amp_mean))
        });

    let denominator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, _)| {
            acc + ((idx as f64 * bin_transform) - freq_mean).powi(2)
        });

    numerator / denominator
}

/// Calculates the spectral spread of a signal.
///
/// The spectral spread is the "instantaneous bandwidth" of the spectrum. It
/// can be used as an indication of the dominance of a tone. As tones converge,
/// the spectral spread decreases and it increases as tones diverge.
pub fn spectral_spread(signal: &[f64]) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let mu_1 = spectral_centroid(signal);

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + ((idx as f64 - mu_1).powi(2) * val)
        });

    let denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    (numerator / denominator).sqrt()
}

/// Calculates the zero crossing rate of a signal.
///
/// The zero crossing rate is the rate of sign changes in a signal, e.g. positive to zero to
/// negative and vice-versa. This can be used to detect pitch or percussive sounds.
pub fn zcr(signal: &[f64]) -> f64 {
    // The accumulator for the fold is a tuple starting at zero.
    let zcr = signal.iter().fold((0_f64, 0_f64), |acc, x| {
        (
            // In order to compare the next number to this one, the first element in the
            // tuple is set to be the current element from the signal. The second element in the
            // tuple is the actual accumulator of the zero crossings, as it is only incremented if
            // the sign of the current sample changes.
            *x,
            acc.1
                + if acc.0.signum() != x.signum() {
                    1_f64
                } else {
                    0_f64
                },
        )
    });

    zcr.1
}

// Calculates the spectral centroid and spread in order to reduce redundant calculation.
fn centroid_and_spread(amp_spectrum: &[f64]) -> (f64, f64) {
    let mu_1_numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| acc + (idx as f64 * val));

    let mu_1_denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    let mu_1 = mu_1_numerator / mu_1_denominator;

    let mu_2_numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + ((idx as f64 - mu_1).powi(2) * val)
        });

    let mu_2_denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    let mu_2 = (mu_2_numerator / mu_2_denominator).sqrt();

    (mu_1, mu_2)
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
