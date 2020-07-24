//! Analysis of signal with respect to frequency.
//!
//! Spectral features, also known as frequency-domain features, are obtained by converting a time-based
//! signal into the frequency domain through the use of the Fourier transform. These features can
//! be used in combination with one another to identify timbre, pitch, etc.

/// Calculates the spectral bandwidth of a signal.
///
/// Spectral bandwidth is the difference between the upper and lower frequencies in a continuous
/// band of frequencies.
pub fn bandwidth(amp_spectrum: &[f64], sampling_rate: Option<f64>) -> f64 {
    let samp_rate = match sampling_rate {
        Some(sr) => sr,
        None => 44100_f64,
    };
    let bin_transform = (samp_rate * amp_spectrum.len() as f64) / 2_f64;

    let centroid = centroid(amp_spectrum, sampling_rate);
    amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + val * ((idx as f64 * bin_transform) - centroid).powi(2)
        })
        .sqrt()
}

/// Calculates the spectral centroid of a signal.
///
/// The spectral centroid represents the "center of gravity" for a spectrum.
/// It is often used to determine the timbre (perceived brightness) of a sound.
pub fn centroid(amp_spectrum: &[f64], sampling_rate: Option<f64>) -> f64 {
    let samp_rate = match sampling_rate {
        Some(sr) => sr,
        None => 44100_f64,
    };
    let bin_transform = (samp_rate * amp_spectrum.len() as f64) / 2_f64;

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + ((idx as f64 * bin_transform) * val)
        });

    let denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    numerator / denominator
}

/// Calculates the spectral crest of a signal.
///
/// Spectral crest can be used to determine the peakiness of a spectrum. A
/// higher spectral crest indicates more tonality, while a lower spectral
/// crest denotes more noise.
pub fn crest(amp_spectrum: &[f64], _sampling_rate: Option<f64>) -> f64 {
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
pub fn decrease(amp_spectrum: &[f64], _sampling_rate: Option<f64>) -> f64 {
    let (s_b1, amp_spectrum) = amp_spectrum.split_at(1);

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + ((val - s_b1[0]) / (idx as f64 - 1_f64))
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
pub fn entropy(amp_spectrum: &[f64], _sampling_rate: Option<f64>) -> f64 {
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
pub fn flatness(amp_spectrum: &[f64], _sampling_rate: Option<f64>) -> f64 {
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
pub fn flux(amp_spectrum: &[f64], _sampling_rate: Option<f64>) -> f64 {
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
pub fn kurtosis(amp_spectrum: &[f64], sampling_rate: Option<f64>) -> f64 {
    let (mu_1, mu_2) = centroid_and_spread(&amp_spectrum, sampling_rate);
    let samp_rate = match sampling_rate {
        Some(sr) => sr,
        None => 44100_f64,
    };
    let bin_transform = (samp_rate * amp_spectrum.len() as f64) / 2_f64;

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + (((idx as f64 * bin_transform) - mu_1).powi(4) * val)
        });

    let denominator = mu_2.powi(4) * amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    numerator / denominator
}

/// Calculates the spectral rolloff point of a signal.
///
/// The spectral rolloff point is the frequency bin under which a given percentage of the total
/// energy in a spectrum exists. It can be used to distinguish unique types of audio in many
/// different situations.
pub fn rolloff(
    amp_spectrum: &[f64],
    sampling_rate: Option<f64>,
    energy_threshold: Option<f64>,
) -> f64 {
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
pub fn skewness(amp_spectrum: &[f64], sampling_rate: Option<f64>) -> f64 {
    let (mu_1, mu_2) = centroid_and_spread(&amp_spectrum, sampling_rate);
    let samp_rate = match sampling_rate {
        Some(sr) => sr,
        None => 44100_f64,
    };
    let bin_transform = (samp_rate * amp_spectrum.len() as f64) / 2_f64;

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + (((idx as f64 * bin_transform) - mu_1).powi(3) * val)
        });

    let denominator = mu_2.powi(3) * amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    numerator / denominator
}

/// Calculates the spectral slope of a signal.
///
/// Spectral energy measures the amount of decrease of the spectrum. It is most pronounced when the
/// energy in the lower formants is much greater than the energy of the higer formants.
pub fn slope(amp_spectrum: &[f64], sampling_rate: Option<f64>) -> f64 {
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
pub fn spread(amp_spectrum: &[f64], sampling_rate: Option<f64>) -> f64 {
    let mu_1 = centroid(amp_spectrum, sampling_rate);
    let samp_rate = match sampling_rate {
        Some(sr) => sr,
        None => 44100_f64,
    };
    let bin_transform = (samp_rate * amp_spectrum.len() as f64) / 2_f64;

    let numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + (((idx as f64 * bin_transform) - mu_1).powi(2) * val)
        });

    let denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    (numerator / denominator).sqrt()
}

// Calculates the spectral centroid and spread in order to reduce redundant calculation.
fn centroid_and_spread(amp_spectrum: &[f64], sampling_rate: Option<f64>) -> (f64, f64) {
    let samp_rate = match sampling_rate {
        Some(sr) => sr,
        None => 44100_f64,
    };
    let bin_transform = (samp_rate * amp_spectrum.len() as f64) / 2_f64;

    let mu_1_numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + ((idx as f64 * bin_transform) * val)
        });

    let mu_1_denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    let mu_1 = mu_1_numerator / mu_1_denominator;

    let mu_2_numerator = amp_spectrum
        .iter()
        .enumerate()
        .fold(0_f64, |acc, (idx, val)| {
            acc + (((idx as f64 * bin_transform) - mu_1).powi(2) * val)
        });

    let mu_2_denominator = amp_spectrum.iter().fold(0_f64, |acc, x| acc + x);

    let mu_2 = (mu_2_numerator / mu_2_denominator).sqrt();

    (mu_1, mu_2)
}
