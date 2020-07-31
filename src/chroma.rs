//! Analysis of signal with respect to pitch classes.
//!
//! Chroma-based features can be used in analysing music. These features show a high degree of
//! robustness to variations in timbre. In combination with other features, chroma features can be
//! used in structure analysis as well as content-based retrieval.

/// Creates bank of chroma filters.
///
/// There are several options that can be passed to this function in order to make analysis of a
/// signal more specific to a priori knowledge, e.g. using a 24-tone equal temperament system
/// instead of the incredibly common 12-tone system used in Western music.
///
/// Optional values explained:
/// * num_chroma_bands: number of pitch classes across which frequencies will be distributed
/// (default: 12)
/// * sampling_rate: sampling rate that the signal at which the signal was record (default:
/// 44100_f64)
/// * standard_pitch: reference frequency from which pitch class frequencies will be calculated
/// (default: 440_f64)
/// * center_octave: along with octave_width, specifies a dominance window; gaussian weighting
/// is centered on center_octave (default: 5)
/// * octave_width: along with center_octave, specifies a dominance window; gaussian weighting will
/// use half-width of size octave_width (default: 2)
/// * base_c: if enabled, chromagram starts at pitch class C (default: true)
pub fn create_chroma_filter_bank(
    buffer_length: usize,
    num_chroma_bands: Option<usize>,
    sampling_rate: Option<f64>,
    standard_pitch: Option<f64>,
    center_octave: Option<usize>,
    octave_width: Option<usize>,
    base_c: Option<bool>,
) -> Vec<Vec<f64>> {
    // Default to 12-TET
    let num_chroma = match num_chroma_bands {
        Some(i) => i,
        None => 12,
    };

    let samp_rate = match sampling_rate {
        Some(sr) => sr,
        None => 44100_f64,
    };

    let standard_pitch = match standard_pitch {
        Some(freq) => freq,
        None => 440_f64,
    };

    let center_octave = match center_octave {
        Some(co) => co,
        None => 5,
    };

    let octave_width = match octave_width {
        Some(ow) => ow,
        None => 2,
    };

    let base_c = match base_c {
        Some(b) => b,
        None => true,
    };

    let num_output_bins = ((buffer_length as f64 / 2_f64).floor() + 1_f64) as u32;

    let freqencies = linspace(0_f64, samp_rate, buffer_length, false);

    let mut freq_bins: Vec<f64> = freqencies
        .iter()
        .map(|freq| num_chroma as f64 * (16_f64 * freq / standard_pitch).log2())
        .collect();

    // Set a value for the 0 Hz bin that is 1.5 octaves below bin 1
    // (so chroma is 50% rotated from bin 1, and bin width is broad)
    freq_bins[0] = freq_bins[1] - 1.5_f64 * num_chroma as f64;

    let mut mut_bin_width_bins = freq_bins
        .iter()
        .enumerate()
        .filter(|&(idx, _)| idx > 0)
        .map(|(idx, val)| (val - freq_bins[idx]).max(1_f64))
        .collect::<Vec<f64>>();

    mut_bin_width_bins.push(1_f64);

    // Derefencing to avoid FnMut closure error during weights construction
    let bin_width_bins = &*mut_bin_width_bins;

    let half_num_chroma = (num_chroma as f64 / 2_f64).floor();

    let filter_peaks: Vec<Vec<f64>> = (0..num_chroma)
        .map(|i| {
            freq_bins
                .iter()
                .map(move |freq| {
                    ((10_f64 * num_chroma as f64 + half_num_chroma + freq - i as f64)
                        % num_chroma as f64)
                        - half_num_chroma
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    let weights: Vec<Vec<f64>> = filter_peaks
        .iter()
        .map(|v| {
            v.iter()
                .enumerate()
                .map(move |(j, _)| (0.5_f64 * (2_f64 * v[j] / bin_width_bins[j]).powf(2_f64)).exp())
                .collect::<Vec<f64>>()
        })
        .collect();

    let column_normalized_weights: Vec<Vec<f64>> = normalize_2d_vec(weights);

    let octave_weights: Vec<f64> = freq_bins
        .into_iter()
        .map(|val| {
            (-0.5_f64
                * ((val / num_chroma as f64 - center_octave as f64) / octave_width as f64)
                    .powf(2_f64))
            .exp()
        })
        .collect();

    // Yeah, the name sucks...
    let mut reweighted_weights: Vec<Vec<f64>> = column_normalized_weights
        .into_iter()
        .map(|column| {
            column
                .iter()
                .enumerate()
                .map(|(idx, val)| val * octave_weights[idx])
                .collect()
        })
        .collect();

    let final_weights = if base_c {
        let mut second = reweighted_weights.split_off(3);
        second.append(&mut reweighted_weights);
        second
    } else {
        reweighted_weights
    };

    final_weights
        .iter()
        .map(|column| column[0..num_output_bins as usize].to_vec())
        .collect()
}

fn normalize_2d_vec(array: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut transpose: Vec<Vec<f64>> = Vec::new();
    for i in 0..array.len() {
        let mut column = Vec::new();
        for row in &array {
            column.push(row[i]);
        }
        transpose.push(column);
    }

    let norm_factors: Vec<f64> = transpose
        .iter()
        .map(|column| column.iter().fold(0_f64, |acc, val| acc + val.powf(2_f64)))
        .map(|sum| sum.sqrt())
        .collect();

    let mut normalized_array: Vec<Vec<f64>> = Vec::new();
    let mut idx: usize = 0;
    for row in &array {
        let mut norm_row = Vec::new();
        for x in row {
            norm_row.push(x / norm_factors[idx]);
            idx += 1;
        }
        normalized_array.push(norm_row);
        idx = 0;
    }

    normalized_array
}

fn linspace(start: f64, stop: f64, length: usize, incl_end: bool) -> Vec<f64> {
    if start != stop {
        // By default, inclusive of stop
        let dx: f64 = if incl_end {
            (stop - start) / (length - 1) as f64
        } else {
            (stop - start) / length as f64
        };

        let mut linspace: Vec<f64> = Vec::with_capacity(length);

        for i in 0..length {
            let jump: f64 = start + i as f64 * dx;
            linspace.push(jump);
        }

        linspace
    } else {
        // If start == stop, return 1-element vector containing start
        let mut linspace: Vec<f64> = Vec::with_capacity(length);
        linspace.push(start);
        linspace
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_2d_normalization() {
        let row1 = [-8_f64, 4_f64, -2_f64, 1_f64].to_vec();
        let row2 = [-1_f64, 1_f64, -1_f64, 1_f64].to_vec();
        let row3 = [0_f64, 0_f64, 0_f64, 1_f64].to_vec();
        let row4 = [1_f64, 1_f64, 1_f64, 1_f64].to_vec();

        let mut vec_2d = Vec::new();
        vec_2d.push(row1);
        vec_2d.push(row2);
        vec_2d.push(row3);
        vec_2d.push(row4);

        let ex_row1 = [
            -0.9847319278346618,
            0.9428090415820635,
            -0.8164965809277261,
            0.5,
        ]
        .to_vec();
        let ex_row2 = [
            -0.12309149097933272,
            0.23570226039551587,
            -0.4082482904638631,
            0.5,
        ]
        .to_vec();
        let ex_row3 = [0.0, 0.0, 0.0, 0.5].to_vec();
        let ex_row4 = [
            0.12309149097933272,
            0.23570226039551587,
            0.4082482904638631,
            0.5,
        ]
        .to_vec();

        let mut ex_vec_2d = Vec::new();
        ex_vec_2d.push(ex_row1);
        ex_vec_2d.push(ex_row2);
        ex_vec_2d.push(ex_row3);
        ex_vec_2d.push(ex_row4);

        let result = normalize_2d_vec(vec_2d);

        assert_eq!(result, ex_vec_2d);
    }

    #[test]
    fn check_linspace() {
        let v = linspace(0.0, 5.0, 6, true);
        let expected = [0_f64, 1_f64, 2_f64, 3_f64, 4_f64, 5_f64].to_vec();
        assert_eq!(v, expected);
    }
}
