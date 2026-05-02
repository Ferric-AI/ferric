// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;

// --- Rejection sampling with a deterministic observed variable ---
//
// Model:
//   x   ~ Bernoulli(0.5)
//   two_x = 2 * (x as u8)        (deterministic)
//
// Observation: two_x = 2  →  forces x = true (2 * 1 = 2)
// Query:       x
//
// Every accepted sample must have x = true because two_x = 2 is only
// achievable when x = true.

#[test]
fn rejection_sampling_deterministic_observation() {
    make_model! {
        mod det_obs_reject;
        use ferric::distributions::Bernoulli;

        let x : bool ~ Bernoulli::new(0.5);
        let two_x : u8 = 2u8 * x as u8;

        observe two_x;
        query x;
    };

    let model = det_obs_reject::Model { two_x: 2 };
    let num_samples = 1000;
    for s in model.sample_iter().take(num_samples) {
        assert!(s.x, "x must be true when two_x is observed to be 2");
    }
}

// --- Weighted sampling with a deterministic queried variable ---
//
// Model:
//   x     ~ Bernoulli(0.5)
//   two_x = 2 * (x as u8)        (deterministic, queried)
//
// Observation: x = true  (stochastic, valid for weighted sampling)
// Query:       two_x
//
// Because x is pinned to true, two_x must evaluate to 2 in every sample.

#[test]
fn weighted_sampling_deterministic_query() {
    make_model! {
        mod det_query_weighted;
        use ferric::distributions::Bernoulli;

        let x : bool ~ Bernoulli::new(0.5);
        let two_x : u8 = 2u8 * x as u8;

        observe x;
        query two_x;
    };

    let model = det_query_weighted::Model { x: true };
    let num_samples = 1000;
    for ws in model.weighted_sample_iter().take(num_samples) {
        assert_eq!(
            ws.sample.two_x, 2,
            "two_x must be 2 when x is observed to be true"
        );
    }
}
