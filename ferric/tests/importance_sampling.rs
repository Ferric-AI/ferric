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
        proposal_dist: Normal,
    }

    impl scalar_importance::Proposer<rand::rngs::ThreadRng> for ProposalFromObservation {
        fn new(data: &scalar_importance::ObservedData) -> Self {
            Self {
                proposal_dist: Normal::new(data.b / 5.0, 4.0).unwrap(),
            }
        }

        fn propose(&mut self, rng: &mut rand::rngs::ThreadRng) -> scalar_importance::Proposal {
            let a = self.proposal_dist.sample(rng);
            let log_prob =
                <Normal as Distribution<rand::rngs::ThreadRng>>::log_prob(&self.proposal_dist, &a);
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
        .importance_sampler::<ProposalFromObservation>()
        .take(num_samples)
    {
        values.push(ws.sample.a);
        log_weights.push(ws.log_weight);
    }

    let post_mean = ferric::weighted_mean(&values, &log_weights);
    let post_std = ferric::weighted_std(&values, &log_weights);
    let effective_sample_size = ferric::effective_sample_size(&log_weights);

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
    assert!(effective_sample_size > 1.0);
    assert!(effective_sample_size <= num_samples as f64);
}

#[test]
fn indexed_deterministic_observation_filters_user_proposal() {
    make_model! {
        name indexed_deterministic_importance;
        use ferric::distributions::Bernoulli;

        const n : u64;
        let latent[i of n] : bool ~ Bernoulli::new(0.5);
        let copied[i of n] : bool = latent[i];

        observe copied;
        query latent;
    };

    struct MatchingProposal {
        proposal: Vec<Option<bool>>,
    }

    impl indexed_deterministic_importance::Proposer<rand::rngs::ThreadRng> for MatchingProposal {
        fn new(data: &indexed_deterministic_importance::ObservedData) -> Self {
            Self {
                proposal: data
                    .copied
                    .iter()
                    .map(|observed| Some(observed.unwrap_or(true)))
                    .collect(),
            }
        }

        fn propose(
            &mut self,
            _rng: &mut rand::rngs::ThreadRng,
        ) -> indexed_deterministic_importance::Proposal {
            let mut proposal = indexed_deterministic_importance::Proposal::new(0.0);
            proposal.latent = self.proposal.clone();
            proposal
        }
    }

    let model = indexed_deterministic_importance::Model {
        n: 3,
        copied: vec![Some(true), None, Some(false)],
    };

    let sample = model
        .importance_sampler::<MatchingProposal>()
        .next()
        .unwrap();

    assert_eq!(sample.sample.latent, vec![true, true, false]);
    assert!(sample.log_weight.is_finite());
}
