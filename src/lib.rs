//! # audisee
//!
//! This library provides several methods for extracting features from audio
//! data. It provides a foundation upon which higher-level audio information
//! systems can be built.

use num_complex::Complex;
use rustfft::FFTplanner;

// TODO: Abstract this away by enabling custom sampling rates.
// Common sampling rate.
const SAMPLING_RATE: f64 = 44100_f64;

/// Returns the amplitude spectrum of a signal block.
///
/// In the amplitude spectrum, each index is a frequency bin; each bin is a
/// range of frequencies. Each element in the spectrum is the amplitude
/// corresponding to that frequency bin. All of the spectral features make
/// use of the frequencies calculated in the spectrum as well as their
/// respective magnitudes.
pub fn amp_spectrum(signal: &Vec<f64>) -> Vec<f64> {
    let signal_length = signal.len();
    let fft = FFTplanner::new(false).plan_fft(signal_length);

    let mut complex_signal = signal
        .iter()
        .map(|x| Complex::new(*x, 0f64))
        .collect::<Vec<Complex<f64>>>();

    let mut spectrum = vec![Complex::new(0_f64, 0_f64); signal_length];

    fft.process(&mut complex_signal, &mut spectrum);

    let amp_spectrum: Vec<f64> = spectrum
        .iter()
        .take(spectrum.len() / 2)
        .map(|bin| (bin.re.powi(2) + bin.im.powi(2)).sqrt())
        .collect();

    amp_spectrum
}

/// Returns the power spectrum of a signal block.
///
/// This differs from an amplitude spectrum as it emphasizes the differences
/// between frequency bins.
pub fn power_spectrum(signal: &Vec<f64>) -> Vec<f64> {
    let amp_spectrum = amp_spectrum(signal);

    let power_spectrum = amp_spectrum.iter().map(|bin| bin.powi(2)).collect();

    power_spectrum
}

pub fn chroma() {
    todo!();
}

/// Calculates the energy of a signal.
///
/// This can be used to determine the loudness of a signal.
pub fn energy(signal: &Vec<f64>) -> f64 {
    let energy = signal
        .iter()
        .fold(0_f64, |acc, &sample| acc + sample.abs().powi(2));

    energy
}

/// Returns the output of the fast Fourier transform using a signal as input.
///
/// The output of the FFT is a complex vector. It contains information of the
/// frequencies present in a signal, namely the magnitudes and phases of each
/// frequency. It is exposed here in order to enable operations that are not
/// included in the library at this time.
pub fn fft(signal: &Vec<f64>) -> Vec<num_complex::Complex<f64>> {
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

pub fn mfcc() {
    todo!();
}

/// Calculates the root mean square of a signal.
///
/// This can be used to determine the loudness of a signal.
pub fn rms(signal: &Vec<f64>) -> f64 {
    let sum = signal
        .iter()
        .fold(0_f64, |acc, &sample| acc + sample.powi(2));

    let mean = sum / signal.len() as f64;

    mean.sqrt()
}

pub fn stft() {
    todo!();
}

pub fn spectral_bandwidth(signal: &Vec<f64>) -> f64 {
    todo!();
}

/// Calculates the spectral centroid of a signal.
///
/// The spectral centroid represents the "center of gravity" for a spectrum.
/// It is often used to determine the timbre (perceived brightness) of a sound.
pub fn spectral_centroid(signal: &Vec<f64>) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| acc + (idx as f64 * val));

    let denominator = signal.iter().fold(0_f64, |acc, x| acc + x);

    numerator / denominator
}

/// Calculates the spectral crest of a signal.
///
/// Spectral crest can be used to determine the peakiness of a spectrum. A
/// higher spectral crest indicates more tonality, while a lower spectral
/// crest denotes more noise.
pub fn spectral_crest(signal: &Vec<f64>) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let numerator = amp_spectrum
        .iter()
        .fold(std::f64::NEG_INFINITY, |a, b| a.max(*b));

    let denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x) / amp_spectrum.len() as f64;

    numerator / denominator
}

/// Calculates the spectral entropy of a signal.
///
/// Spectral entropy measures the peakiness of a spectrum. As entropy is a
/// measure of disorder, it can be used to differentiate between types of
/// sound that have different expectations of "order", e.g. speech vs.
/// music with multiple instruments.
pub fn spectral_entropy(signal: &Vec<f64>) -> f64 {
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
pub fn spectral_flatness(signal: &Vec<f64>) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let numerator = amp_spectrum
        .iter()
        .fold(0_f64, |acc, x| acc * x)
        .powf(1_f64 / amp_spectrum.len() as f64);

    let denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x) / amp_spectrum.len() as f64;

    numerator / denominator
}

pub fn spectral_flux() {
    todo!();
}

/// Calculates the spectral kurtosis of a signal.
///
/// Spectral kurtosis measures the flatness of a spectrum near its centroid
/// ("center of gravity"). It can also be used to measure the peakiness of
/// a spectrum as well.
pub fn spectral_kurtosis(signal: &Vec<f64>) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let mu_1 = spectral_centroid(signal);
    let mu_2 = spectral_spread(signal);

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + ((idx as f64 - mu_1).powi(4) * val)
        });

    let denominator = mu_2.powi(4) * amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    numerator / denominator
}

pub fn spectral_rolloff() {
    todo!();
}

/// Calculates the spectral skewness of a signal.
///
/// Spectral skewness measures symmetry around the centroid.
pub fn spectral_skewness(signal: &Vec<f64>) -> f64 {
    let amp_spectrum = amp_spectrum(signal);
    let mu_1 = spectral_centroid(signal);
    let mu_2 = spectral_spread(signal);

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + ((idx as f64 - mu_1).powi(3) * val)
        });

    let denominator = mu_2.powi(3) * amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    numerator / denominator
}

pub fn spectral_slope() {
    todo!();
}

/// Calculates the spectral spread of a signal.
///
/// The spectral spread is the "instantaneous bandwidth" of the spectrum. It
/// can be used as an indication of the dominance of a tone. As tones converge,
/// the spectral spread decreases and it increases as tones diverge.
pub fn spectral_spread(signal: &Vec<f64>) -> f64 {
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

pub fn spectrogram() {
    todo!();
}

/// Calculates the zero crossing rate of a signal.
///
/// The zero crossing rate is the rate of sign changes in a signal, e.g. positive to zero to
/// negative and vice-versa. This can be used to detect pitch or percussive sounds.
pub fn zcr(signal: &Vec<f64>) -> f64 {
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

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}