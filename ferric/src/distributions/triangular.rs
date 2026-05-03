// Copyright 2022 The Ferric AI Project Developers

use rand::Rng;
use rand_distr::Distribution as Distribution2;
use rand_distr::Triangular as Triangular2;
use rand_distr::TriangularError;

use crate::distributions::Distribution;

/// Triangular distribution over $[a, b]$.
///
/// The PDF is
///
/// $$p(x \mid a, b, c) = \begin{cases}
///     \dfrac{2(x-a)}{(b-a)(c-a)} & a \le x \le c \\
///     \dfrac{2(b-x)}{(b-a)(b-c)} & c < x \le b
/// \end{cases}$$
///
/// where $a$ is the lower limit, $b$ is the upper limit, and $c$ is
/// the mode.
///
/// See [Triangular distribution](https://en.wikipedia.org/wiki/Triangular_distribution)
/// on Wikipedia for further details.
///
/// # Examples
///
/// ```
/// use ferric::distributions::{Distribution, Triangular};
/// use rand::thread_rng;
///
/// let dist = Triangular::new(0.0, 4.0, 1.0).unwrap();
/// let x: f64 = dist.sample(&mut thread_rng());
/// println!("sample = {:.4}", x);
/// ```
pub struct Triangular {
    a: f64,
    b: f64,
    c: f64,
}

impl Triangular {
    /// Construct a Triangular distribution with lower limit `a`, upper limit
    /// `b`, and mode `c`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the parameters are invalid (requires $a < b$ and
    /// $a \le c \le b$).
    pub fn new(a: f64, b: f64, c: f64) -> Result<Triangular, String> {
        Triangular2::new(a, b, c)
            .map(|_| Triangular { a, b, c })
            .map_err(|e: TriangularError| format!("Triangular: {:?}", e))
    }
}

impl<R: Rng + ?Sized> Distribution<R> for Triangular {
    type Domain = f64;

    fn sample(&self, rng: &mut R) -> f64 {
        Triangular2::new(self.a, self.b, self.c)
            .unwrap()
            .sample(rng)
    }

    /// Returns the log of the piecewise-linear PDF, or $-\infty$ outside
    /// $[a, b]$.
    fn log_prob(&self, x: &f64) -> f64 {
        let x = *x;
        let range = self.b - self.a;
        if x < self.a || x > self.b {
            return f64::NEG_INFINITY;
        }
        if x <= self.c {
            (2.0 * (x - self.a) / (range * (self.c - self.a))).ln()
        } else {
            (2.0 * (self.b - x) / (range * (self.b - self.c))).ln()
        }
    }

    fn is_discrete(&self) -> bool {
        false
    }
}

impl std::fmt::Display for Triangular {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Triangular {{ a = {}, b = {}, c = {} }}",
            self.a, self.b, self.c
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;
    use rand::thread_rng;

    #[test]
    fn triangular_sample() {
        let mut rng = thread_rng();
        let a = 0.0f64;
        let b = 4.0f64;
        let c = 1.0f64;
        let dist = Triangular::new(a, b, c).unwrap();
        println!("dist = {}", dist);
        let trials = 100_000;
        let mut total = 0.0f64;
        for _ in 0..trials {
            total += dist.sample(&mut rng);
        }
        let empirical_mean = total / trials as f64;
        // Mean = (a + b + c) / 3
        let expected_mean = (a + b + c) / 3.0;
        // Variance = (a^2 + b^2 + c^2 - ab - ac - bc) / 18
        let variance = (a * a + b * b + c * c - a * b - a * c - b * c) / 18.0;
        let std = variance.sqrt();
        let err = 5.0 * std / (trials as f64).sqrt();
        assert!((empirical_mean - expected_mean).abs() < err);
    }

    #[test]
    fn triangular_log_prob() {
        // Triangular(0, 2, 1) at x=1 (the mode): 2(1-0)/((2-0)*(1-0)) = 1
        // log_prob = ln(1) = 0
        let dist = Triangular::new(0.0, 2.0, 1.0).unwrap();
        let lp = <Triangular as Distribution<ThreadRng>>::log_prob(&dist, &1.0);
        assert!((lp - 0.0f64).abs() < 1e-10);

        // x outside [a, b] → NEG_INFINITY
        let lp_out = <Triangular as Distribution<ThreadRng>>::log_prob(&dist, &3.0);
        assert_eq!(lp_out, f64::NEG_INFINITY);

        let lp_right = <Triangular as Distribution<ThreadRng>>::log_prob(&dist, &1.5);
        assert!((lp_right - (0.5_f64).ln()).abs() < 1e-10);

        assert!(!<Triangular as Distribution<ThreadRng>>::is_discrete(&dist));
    }

    #[test]
    fn triangular_display() {
        let dist = Triangular::new(0.0, 4.0, 1.0).unwrap();
        let s = format!("{}", dist);
        assert!(s.contains("Triangular"), "missing type name: {}", s);
    }

    #[test]
    fn triangular_invalid() {
        assert!(Triangular::new(2.0, 1.0, 1.5).is_err());
    }
}
