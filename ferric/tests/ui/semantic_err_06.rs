// Copyright 2022 The Ferric AI Project Developers
//
// Negative test: weighted_sample_iter is not available when a deterministic
// variable is observed, because there is no distribution from which to
// compute the log-likelihood of the observation.
use ferric::make_model;

make_model! {
    mod det_weighted_obs;
    use ferric::distributions::Bernoulli;

    let x : bool ~ Bernoulli::new(0.5);
    let two_x : u8 = 2u8 * x as u8;

    observe two_x;
    query x;
}

fn main() {
    let model = det_weighted_obs::Model { two_x: 2 };
    let _ = model.weighted_sample_iter();
}
