// Copyright 2022 The Ferric AI Project Developers
use ferric::distributions::{Distribution, Normal};
use ferric::make_model;

#[test]
fn scalar_importance_sampler_uses_user_proposal() {
    make_model! {
        name scalar_importance;
        use ferric::distributions::Normal;

        let a : f64 ~ Normal::new(0.0, 1.0);
        let b : f64 ~ Normal::new(5.0 * a, 3.0);

        observe b;
        query a;
    };

    struct ProposalFromObservation {
        proposal_dist: Option<Normal>,
    }

    impl scalar_importance::Proposer<rand::rngs::ThreadRng> for ProposalFromObservation {
        fn initialize(&mut self, data: &scalar_importance::ObservedData) {
            self.proposal_dist = Some(Normal::new(data.b / 5.0, 4.0).unwrap());
        }

        fn propose(&mut self, rng: &mut rand::rngs::ThreadRng) -> scalar_importance::Proposal {
            let proposal_dist = self
                .proposal_dist
                .as_ref()
                .expect("Ferric initializes proposers before sampling");
            let a = proposal_dist.sample(rng);
            let log_prob =
                <Normal as Distribution<rand::rngs::ThreadRng>>::log_prob(proposal_dist, &a);
            let mut proposal = scalar_importance::Proposal::new(log_prob);
            proposal.a = Some(a);
            proposal
        }
    }

    let model = scalar_importance::Model { b: 20.0 };
    let num_samples = 100000;
    let mut values = Vec::with_capacity(num_samples);
    let mut log_weights = Vec::with_capacity(num_samples);

    for ws in model
        .importance_sampler(ProposalFromObservation {
            proposal_dist: None,
        })
        .take(num_samples)
    {
        values.push(ws.sample.a);
        log_weights.push(ws.log_weight);
    }

    let post_mean = ferric::weighted_mean(&values, &log_weights);
    let post_std = ferric::weighted_std(&values, &log_weights);

    // A ~ N(0, 1), B | A ~ N(5A, 3^2), observed B = 20.
    // Gaussian conditioning gives:
    //   Var(A | B) = 1 / (1 / 1^2 + 5^2 / 3^2) = 9 / 34
    //   E(A | B)  = Var(A | B) * (5 * 20 / 3^2) = 100 / 34
    let expected_mean = 100.0 / 34.0;
    let expected_std = (9.0f64 / 34.0).sqrt();

    assert!(
        (post_mean - expected_mean).abs() < 0.08,
        "posterior mean {post_mean} not close to {expected_mean}"
    );
    assert!(
        (post_std - expected_std).abs() < 0.08,
        "posterior std {post_std} not close to {expected_std}"
    );
}
