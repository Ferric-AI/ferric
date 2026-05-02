// Copyright 2022 The Ferric AI Project Developers
use ferric::make_model;

// Regression test: observed variables must not be re-sampled during reset.
//
// Model:
//   var1 ~ Beta(1.0, 1.0)
//   var2 ~ Bernoulli(var1)
//
// Observation: var1 = 1.0  (probability of success is 1.0)
// Query:       var2
//
// Because var1 is clamped to 1.0, Bernoulli(1.0) always fires, so every
// sample of var2 must be `true`.  If reset() incorrectly sets var1 back to
// Unknown and the eval path re-samples it from the prior, some samples will
// draw var1 < 1.0 and produce var2 = false, causing the assertion to fail.

#[test]
fn observed_var_not_resampled() {
    make_model! {
        mod observed_var_not_resampled;
        use ferric::distributions::Beta;
        use ferric::distributions::Bernoulli;

        let var1 : f64 ~ Beta::new(1.0, 1.0);
        let var2 : bool ~ Bernoulli::new(var1);

        observe var1;
        query var2;
    };

    let model = observed_var_not_resampled::Model { var1: 1.0 };

    let num_samples = 10000;
    for ws in model.weighted_sample_iter().take(num_samples) {
        assert!(
            ws.sample.var2,
            "var2 must always be true when var1 is observed to be 1.0"
        );
    }
}
