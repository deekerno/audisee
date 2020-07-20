//! Analysis of signal with respect to time.

/// Calculates the energy of a signal.
///
/// This can be used to determine the loudness of a signal.
pub fn energy(signal: &[f64]) -> f64 {
    signal
        .iter()
        .fold(0_f64, |acc, &sample| acc + sample.abs().powi(2))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms() {
        let rms = rms(crate::utils::TEST_SIGNAL);
        assert_eq!(rms, 0.5602403001749346);
    }

    #[test]
    fn test_zcr() {
        let zcr = zcr(crate::utils::TEST_SIGNAL);
        assert_eq!(zcr, 58.0);
    }

    #[test]
    fn test_energy() {
        let energy = energy(crate::utils::TEST_SIGNAL);
        assert_eq!(energy, 40.175256824332905);
    }
}
