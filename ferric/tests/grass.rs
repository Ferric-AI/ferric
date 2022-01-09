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
    let post_rain = (model
        .sample_iter()
        .take(num_samples)
        .map(|x| x.rain as isize)
        .sum::<isize>() as f64)
        / (num_samples as f64);
    println!("post_rain is {}", post_rain);
    assert!(post_rain > (ans - err) && post_rain < (ans + err));
}
