// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;
use std::time::Instant;

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

    observe grass_wet;
    query rain;
    query sprinkler;
}

fn main() {
    let model = grass::Model { grass_wet: true };
    let num_samples = 100000;

    // --- Rejection sampling (original method) ---
    let start = Instant::now();
    let mut num_rain = 0usize;
    let mut num_sprinkler = 0usize;
    for sample in model.sample_iter().take(num_samples) {
        if sample.rain {
            num_rain += 1;
        }
        if sample.sprinkler {
            num_sprinkler += 1;
        }
    }
    let reject_elapsed = start.elapsed().as_millis();
    let post_rain_reject = num_rain as f64 / num_samples as f64;
    let post_sprinkler_reject = num_sprinkler as f64 / num_samples as f64;

    // --- Likelihood-weighted sampling (new method) ---
    let start = Instant::now();
    let mut rain_vals = Vec::with_capacity(num_samples);
    let mut sprinkler_vals = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);
    for ws in model.weighted_sample_iter().take(num_samples) {
        rain_vals.push(ws.sample.rain as u8 as f64);
        sprinkler_vals.push(ws.sample.sprinkler as u8 as f64);
        log_weights.push(ws.log_weight);
    }
    let weighted_elapsed = start.elapsed().as_millis();
    let post_rain_weighted = ferric::weighted_mean(&rain_vals, &log_weights);
    let post_sprinkler_weighted = ferric::weighted_mean(&sprinkler_vals, &log_weights);

    println!(
        "rejection   : rain = {:.4} sprinkler = {:.4}  ({} ms, {} samples)",
        post_rain_reject, post_sprinkler_reject, reject_elapsed, num_samples
    );
    println!(
        "lik-weighted: rain = {:.4} sprinkler = {:.4}  ({} ms, {} samples)",
        post_rain_weighted, post_sprinkler_weighted, weighted_elapsed, num_samples
    );
}
