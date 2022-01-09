// Copyright 2022 The Ferric AI Project Developers

use rand::rngs::ThreadRng;
use rand::Rng;

/// A Distribution<Domain=...> defined over a specific domain can be used to generate
/// random values over that domain.
/// For example, the Bernoulli distribution implements Distribution<Domain=bool> and
/// it can be used to generate random booleans.
pub trait Distribution<R = ThreadRng>
where
    R: Rng + ?Sized,
{
    type Domain;
    fn sample(&self, rng: &mut R) -> Self::Domain;
}
