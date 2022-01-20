// Copyright 2022 The Ferric AI Project Developers
use rand_distr::Distribution as Distribution2;
use rand_distr::Poisson as Poisson2;

use crate::distributions::Distribution;
use rand::Rng;

pub struct Poisson {
    /// Probability of success.
    rate: f64,
}

impl Poisson {
    pub fn new(rate: f64) -> Result<Poisson, String> {
        if rate <= 0f64 {
            Err(format! {"Poisson: illegal rate `{}` should be greater than 0", rate})
        } else {
            Ok(Poisson { rate })
        }
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Poisson {
    type Domain = u64;
    fn sample(&self, rng: &mut R) -> u64 {
        Poisson2::new(self.rate).unwrap().sample(rng) as u64
    }
}

impl std::fmt::Display for Poisson {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Poisson {{ rate = {} }}", self.rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn poisson_sample() {
        let mut rng = thread_rng();
        let rate = 2.7f64;
        let dist = Poisson::new(rate).unwrap();
        println!("dist = {}", dist);
        let mut total = 0u64;
        let trials = 10000;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let mean = (total as f64) / (trials as f64);
        let err = 5.0 * (rate / (trials as f64)).sqrt();
        println!("empirical mean is {} 5sigma error is {}", mean, err);
        assert!((mean - 2.7).abs() < err);
    }

    #[test]
    #[should_panic]
    fn poisson_too_low() {
        let _dist = Poisson::new(-0.01).unwrap();
    }
}
