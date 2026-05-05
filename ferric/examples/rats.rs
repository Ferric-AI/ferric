// Copyright 2022 The Ferric AI Project Developers
// Rats weight model example from Gelfand et al's "Bayesian Data Analysis" (3rd ed), section 3.5.2.
// https://chjackson.github.io/openbugsdoc/Examples/Rats.html

use ferric::make_model;

fn centered_day(time: u64) -> f64 {
    match time {
        0 => -14.0,
        1 => -7.0,
        2 => 0.0,
        3 => 7.0,
        4 => 14.0,
        _ => panic!("rats example has five observation times"),
    }
}

make_model! {
    name rats;
    use ferric::distributions::Gamma;
    use ferric::distributions::Normal;
    use super::centered_day;

    const num_rats : u64;
    const num_times : u64;

    let alpha_mean : f64 ~ Normal::new(250.0, 100.0);
    let beta_mean : f64 ~ Normal::new(6.0, 10.0);
    let alpha_std : f64 ~ Gamma::new(2.0, 20.0);
    let beta_std : f64 ~ Gamma::new(2.0, 2.0);
    let obs_std : f64 ~ Gamma::new(2.0, 5.0);

    let alpha[rat of num_rats] : f64 ~ Normal::new(alpha_mean, alpha_std);
    let beta[rat of num_rats] : f64 ~ Normal::new(beta_mean, beta_std);
    let weight[rat of num_rats, time of num_times] : f64 ~ Normal::new(
        alpha[rat] + beta[rat] * centered_day(time),
        obs_std
    );

    observe weight;
    query alpha_mean;
    query beta_mean;
    query alpha_std;
    query beta_std;
    query obs_std;
}

fn observed_weights() -> Vec<Vec<Option<f64>>> {
    [
        [151.0, 199.0, 246.0, 283.0, 320.0],
        [145.0, 199.0, 249.0, 293.0, 354.0],
        [147.0, 214.0, 263.0, 312.0, 328.0],
        [155.0, 200.0, 237.0, 272.0, 297.0],
        [135.0, 188.0, 230.0, 280.0, 323.0],
        [159.0, 210.0, 252.0, 298.0, 331.0],
        [141.0, 189.0, 231.0, 275.0, 305.0],
        [159.0, 201.0, 248.0, 297.0, 338.0],
        [177.0, 236.0, 285.0, 350.0, 376.0],
        [134.0, 182.0, 220.0, 260.0, 296.0],
        [160.0, 208.0, 261.0, 313.0, 352.0],
        [143.0, 188.0, 220.0, 273.0, 314.0],
        [154.0, 200.0, 244.0, 289.0, 325.0],
        [171.0, 221.0, 270.0, 326.0, 358.0],
        [163.0, 216.0, 242.0, 281.0, 312.0],
        [160.0, 207.0, 248.0, 288.0, 324.0],
        [142.0, 187.0, 234.0, 280.0, 316.0],
        [156.0, 203.0, 243.0, 283.0, 317.0],
        [157.0, 212.0, 259.0, 307.0, 336.0],
        [152.0, 203.0, 246.0, 286.0, 321.0],
        [154.0, 205.0, 253.0, 298.0, 334.0],
        [139.0, 190.0, 225.0, 267.0, 302.0],
        [146.0, 191.0, 229.0, 272.0, 302.0],
        [157.0, 211.0, 250.0, 285.0, 323.0],
        [132.0, 185.0, 237.0, 286.0, 331.0],
        [160.0, 207.0, 257.0, 303.0, 345.0],
        [169.0, 216.0, 261.0, 295.0, 333.0],
        [157.0, 205.0, 248.0, 289.0, 316.0],
        [137.0, 180.0, 219.0, 258.0, 291.0],
        [153.0, 200.0, 244.0, 286.0, 324.0],
    ]
    .into_iter()
    .map(|row| row.into_iter().map(Some).collect())
    .collect()
}

fn main() {
    let model = rats::Model {
        num_rats: 30,
        num_times: 5,
        weight: observed_weights(),
    };

    let sample = model.weighted_sample_iter().next().unwrap();
    println!(
        "Gelfand rats model: 30 rats, 5 observations each, one prior draw log_weight = {:.1}",
        sample.log_weight
    );
}
