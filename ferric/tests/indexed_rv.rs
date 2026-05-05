use ferric::make_model;

make_model! {
    name indexed_const;
    use ferric::distributions::Bernoulli;

    const n : u64;
    let flips[flip of n] : bool ~ Bernoulli::new(0.5);

    observe flips;
    query flips;
}

make_model! {
    name indexed_stochastic_bound;
    use ferric::distributions::Bernoulli;
    use ferric::distributions::Poisson;

    const max_n : u64;
    let n : u64 ~ Poisson::new(4.0) max max_n;
    let flips[flip of n] : bool ~ Bernoulli::new(0.5);
    let num_heads : u64 = flips.iter().filter(|&&flip| flip).count() as u64;

    observe num_heads;
    query n;
}

make_model! {
    name indexed_grid;
    use ferric::distributions::Bernoulli;

    const n : u64;
    const m : u64;
    let grid[row of n, col of m] : bool ~ Bernoulli::new(0.5);

    query grid;
}

make_model! {
    name bounded_poisson;
    use ferric::distributions::Poisson;

    const max_n : u64;
    let n : u64 ~ Poisson::new(1.0) max max_n;

    observe n;
    query n;
}

make_model! {
    name indexed_self_loop;

    const n : u64;
    let x[i of n] : bool = if i == 0 { x[i] } else { true };

    query x;
}

#[test]
fn indexed_array_supports_masked_weighted_observations() {
    let model = indexed_const::Model {
        n: 3,
        flips: vec![Some(true), None, Some(false)],
    };

    let weighted = model.weighted_sample_iter().next().unwrap();
    assert_eq!(weighted.sample.flips.len(), 3);
    assert!(weighted.log_weight.is_finite());
}

#[test]
fn stochastic_index_bound_respects_required_maximum() {
    let model = indexed_stochastic_bound::Model {
        max_n: 2,
        num_heads: 1,
    };

    for sample in model.sample_iter().take(128) {
        assert!(sample.n <= 2);
    }
}

#[test]
fn indexed_arrays_can_be_two_dimensional() {
    let model = indexed_grid::Model { n: 2, m: 3 };
    let sample = model.sample_iter().next().unwrap();

    assert_eq!(sample.grid.len(), 2);
    assert!(sample.grid.iter().all(|row| row.len() == 3));
}

#[test]
fn bounded_observation_uses_truncated_probability_for_weighting() {
    let model = bounded_poisson::Model { max_n: 2, n: 2 };
    let weighted = model.weighted_sample_iter().next().unwrap();
    let p0 = (-1.0f64).exp();
    let p1 = (-1.0f64).exp();
    let p2 = 0.5 * (-1.0f64).exp();
    let expected = p2 / (p0 + p1 + p2);

    assert_eq!(weighted.sample.n, 2);
    assert!((weighted.log_weight.exp() - expected).abs() < 1e-12);
}

#[test]
fn indexed_self_loop_reports_dependency_loop() {
    let model = indexed_self_loop::Model { n: 1 };
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = model.sample_iter().next();
    }))
    .expect_err("indexed self-loop should panic");

    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(message.contains("Ferric dependency loop"));
    assert!(message.contains("x"));
}
