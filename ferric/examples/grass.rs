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
    let mut num_rain = 0;
    let mut num_sprinkler = 0;
    let num_samples = 100000;
    let start = Instant::now();
    for sample in model.sample_iter().take(num_samples) {
        if sample.rain {
            num_rain += 1;
        }
        if sample.sprinkler {
            num_sprinkler += 1;
        }
    }
    let num_samples = num_samples as f64;
    println!(
        "posterior: rain = {} sprinkler = {}. Elapsed {} millisec for {} samples",
        (num_rain as f64) / num_samples,
        (num_sprinkler as f64) / num_samples,
        start.elapsed().as_millis(),
        num_samples,
    );
}
