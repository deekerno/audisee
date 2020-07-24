//! Utility crate.

/// This struct is used to create overlapping frames of a signal.
///
/// Splitting an audio signal into successive non-overlapping
/// frames can lead to smearing of bin magnitudes as it is possible that
/// the signal may not start and end at zero in each frame; this leads to a different phase offset
/// in each frame. A difference in phase offset between frames denotes a deviation in frequency
/// from the bin frequencies. By overlapping the signal's frames, this error in determining the
/// frequency can be reduced.
///
/// ~~~
/// use audisee::utils::OverlappingFrames;
///
/// // Create vector of f64s.
/// let v = vec![
///     1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64, 7_f64, 8_f64, 9_f64, 10_f64, 11_f64, 12_f64,
///     13_f64, 14_f64, 15_f64, 16_f64,
/// ];
///
/// // Create frames of length 4 with 25% overlap between consecutive frames.
/// let of = OverlappingFrames::new(&v, 4, 0.25);
/// let of_vec: Vec<Vec<f64>> = of.into_iter().collect();
/// let expected_frames: Vec<Vec<f64>> = vec![
///     vec![1_f64, 2_f64, 3_f64, 4_f64],
///     vec![4_f64, 5_f64, 6_f64, 7_f64],
///     vec![7_f64, 8_f64, 9_f64, 10_f64],
///     vec![10_f64, 11_f64, 12_f64, 13_f64],
///     vec![13_f64, 14_f64, 15_f64, 16_f64],
/// ];
/// assert_eq!(of_vec, expected_frames);
/// ~~~
#[derive(Debug)]
pub struct OverlappingFrames {
    buffer: Vec<f64>,
    stride: usize,
    frame_size: usize,
}

// TODO: Change return of new function to Result.
impl OverlappingFrames {
    /// Creates a collection of overlapping frames.
    ///
    /// At this time, frame size should be a multiple of four and no greater than half the length
    /// of the signal vector. Available overlap values are 0.25 (25%), 0.50 (50%), and 0.75 (75%).
    pub fn new(buffer: &Vec<f64>, frame_size: usize, overlap: f64) -> Self {
        // Tried to do this with slices, but couldn't get around E0515 after adding padding.
        let padding_needed = buffer.len() % 4;
        let padded_buffer = if padding_needed != 0 {
            let mut buff_clone = buffer.clone();
            let mut padding = vec![0_f64; padding_needed];
            buff_clone.append(&mut padding);
            buff_clone
        } else {
            buffer.clone()
        };

        let stride = frame_size as f64 * (1_f64 - overlap);

        OverlappingFrames {
            buffer: padded_buffer,
            stride: stride as usize,
            frame_size,
        }
    }
}

impl Iterator for OverlappingFrames {
    type Item = Vec<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.frame_size <= self.buffer.len() {
            let subslice = self.buffer[0..self.frame_size].to_vec();

            let advance_amount = std::cmp::min(self.stride, self.buffer.len());
            self.buffer.drain(0..advance_amount);

            Some(subslice)
        } else {
            None
        }
    }
}

pub(crate) const TEST_SIGNAL: &[f64] = &[
    0.35039299166804794,
    0.2264099842027547,
    0.27366499738923755,
    -0.6977997263230118,
    0.2817786441206014,
    0.19262344378614937,
    0.18150065781504088,
    0.3990517280186645,
    -0.20203759856447112,
    0.9454668618147664,
    0.6455063963278889,
    -0.2755419090910798,
    -0.14814210567265773,
    0.4042728070067936,
    -0.11208213268638856,
    0.9523336972668788,
    0.08151983800494866,
    0.7248302096910275,
    0.05052998327919944,
    0.9132568311626783,
    -0.3911660197031206,
    0.7555837923940345,
    -0.4897760891117007,
    0.7837965041822121,
    -0.7014958939854341,
    0.7885738347211002,
    -0.14511079435804186,
    -0.8851790822439902,
    -0.5999578749841521,
    -0.1937554081226791,
    -0.9008575135022117,
    0.07739453230105653,
    0.5470120912905618,
    0.6869990579110339,
    0.7161386143779094,
    0.6290598983899791,
    -0.8373762806409215,
    -0.968152358166908,
    -0.2626372590372674,
    -0.47204743552146944,
    -0.7330923053329608,
    -0.9972673351361943,
    0.8796519435483905,
    -0.08757035021283643,
    0.2021760554768699,
    0.2998351298123212,
    -0.1978473176819211,
    -0.6489911859136335,
    -0.43799779712954123,
    0.5265250385971068,
    -0.5163409091128068,
    -0.05414139410829222,
    -0.562521405748059,
    -0.9794135610502206,
    -0.9138765798758284,
    0.667491848631022,
    0.27957159947219967,
    0.353534859820277,
    -0.039869185527035,
    0.1110169871308293,
    0.682619722088162,
    0.43989736804907276,
    -0.08050694261292879,
    -0.3994768458118658,
    -0.22451812904517743,
    0.5438179657768822,
    0.7393723798786676,
    0.11405625995598667,
    0.5231384940160693,
    0.8314911156378337,
    0.22170374933577852,
    -0.08860701482223732,
    0.3890999860705908,
    -0.0893340555084765,
    0.746341041260647,
    0.25505857240576946,
    -0.9441395883469892,
    -0.7243589535645709,
    -0.40230732169544137,
    0.3460552534562513,
    -0.3512233874425186,
    0.1880659649031342,
    -0.9811453374679484,
    0.6446907577126368,
    -0.7629978940566011,
    -0.8841291911589892,
    0.40785510110548584,
    0.7112418166130778,
    -0.721360470023861,
    -0.14424135685428885,
    -0.25870632004630734,
    -0.1569242190668536,
    0.5374459393728013,
    0.8605531037795786,
    0.4620338522795602,
    0.5715577516180854,
    -0.16307586720650136,
    0.002786614841363999,
    0.7870557050163658,
    -0.30384144497255594,
    -0.16933504349852324,
    0.11994857852371021,
    0.14868873948122685,
    -0.8351550706371436,
    -0.5719420171500165,
    0.47420399829316073,
    0.727096075409646,
    0.23760505083593486,
    -0.4182935876546865,
    -0.48225508821552276,
    0.17649874675017285,
    -0.6869961850955217,
    0.7211454562323887,
    -0.9582735052881874,
    -0.14086722776566152,
    -0.31833227650564844,
    -0.24970897618797538,
    -0.23652584698688406,
    -0.8977547499167211,
    -0.861617843104149,
    -0.7194762202336045,
    0.35728851795987593,
    -0.6616484771622435,
    0.44692908344991844,
    -0.6073006874598104,
    -0.8418763907533826,
    0.03729814000428977,
    0.5889192933707887,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_frames_length() {
        let v = vec![
            1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64, 7_f64, 8_f64, 9_f64, 10_f64, 11_f64, 12_f64,
            13_f64, 14_f64, 15_f64, 16_f64,
        ];
        let of_25 = OverlappingFrames::new(&v, 4, 0.25);
        let of_50 = OverlappingFrames::new(&v, 4, 0.50);
        let of_75 = OverlappingFrames::new(&v, 4, 0.75);
        let of25_vec: Vec<Vec<f64>> = of_25.into_iter().collect();
        let of50_vec: Vec<Vec<f64>> = of_50.into_iter().collect();
        let of75_vec: Vec<Vec<f64>> = of_75.into_iter().collect();
        assert_eq!((of25_vec.len(), of50_vec.len(), of75_vec.len()), (5, 7, 13));
    }

    #[test]
    fn overlapping_frames_chunks() {
        let v = vec![
            1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64, 7_f64, 8_f64, 9_f64, 10_f64, 11_f64, 12_f64,
            13_f64, 14_f64, 15_f64, 16_f64,
        ];
        let of = OverlappingFrames::new(&v, 4, 0.25);
        let of_vec: Vec<Vec<f64>> = of.into_iter().collect();
        let expected_frames: Vec<Vec<f64>> = vec![
            vec![1_f64, 2_f64, 3_f64, 4_f64],
            vec![4_f64, 5_f64, 6_f64, 7_f64],
            vec![7_f64, 8_f64, 9_f64, 10_f64],
            vec![10_f64, 11_f64, 12_f64, 13_f64],
            vec![13_f64, 14_f64, 15_f64, 16_f64],
        ];
        assert_eq!(of_vec, expected_frames);
    }
}
