// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;

// Test the simple Bayesian Network from Wikipedia:
// https://en.wikipedia.org/wiki/Bayesian_network#Example
// P( Rain | Grass Is Wet ) = .3577

#[test]
fn grass() {
    make_model! {
        mod grass;
        use ferric::distributions::Bernoulli;

        let rain : bool ~ Bernoulli::new( 0.2 );

        let sprinkler : bool ~
            if rain {
                Bernoulli::new( 0.01 )
            } else {
                Bernoulli::new( 0.4 )
            };

        let grass_wet : bool ~ Bernoulli::new(
            if sprinkler && rain { 0.99 }
            else if sprinkler && !rain { 0.9 }
            else if !sprinkler && rain { 0.8 }
            else { 0.0 }
        );

        query rain;
        observe grass_wet;
    };

    let model = grass::Model { grass_wet: true };
    let num_samples = 100000;
    let ans = 0.3577f64;
    let err = 5.0 * (ans * (1.0 - ans) / (num_samples as f64)).sqrt(); // 5 sigma error

    // --- Rejection sampling (original method) ---
    let post_rain_reject = (model
        .sample_iter()
        .take(num_samples)
        .map(|s| s.rain as isize)
        .sum::<isize>() as f64)
        / (num_samples as f64);
    println!("rejection    post_rain = {}", post_rain_reject);
    assert!(
        post_rain_reject > (ans - err) && post_rain_reject < (ans + err),
        "rejection post_rain {} outside [{}, {}]",
        post_rain_reject,
        ans - err,
        ans + err
    );

    // --- Likelihood-weighted sampling (new method) ---
    let mut rain_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);
    for ws in model.weighted_sample_iter().take(num_samples) {
        rain_vals.push(ws.sample.rain as u8 as f64);
        log_weights.push(ws.log_weight);
    }
    let post_rain_weighted = ferric::weighted_mean(&rain_vals, &log_weights);
    println!("lik-weighted post_rain = {}", post_rain_weighted);
    assert!(
        post_rain_weighted > (ans - err) && post_rain_weighted < (ans + err),
        "weighted post_rain {} outside [{}, {}]",
        post_rain_weighted,
        ans - err,
        ans + err
    );
}
