// Copyright 2022 The Ferric AI Project Developers
//! Optional value of a Ferric random variable.
//!
//! Type [`FeOption`] represents an potential value for a random variable:
//! every [`FeOption`] is either [`Known`] and contains a value, or [`Null`], or
//! [`Unknown`]. Note that a [`Null`] value is technically a case where the value is known.
//!
#[derive(Copy, Clone)]
pub enum FeOption<T> {
    Null,
    Unknown,
    Known(T),
}

use FeOption::{Known, Null, Unknown};

impl<T> FeOption<T> {
    /// Returns `true` if the FeOption is a [`Null`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ferric::*;
    /// let x: FeOption<u32> = Null;
    /// assert_eq!(x.is_null(), true);
    ///
    /// let x: FeOption<u32> = Known(2);
    /// assert_eq!(x.is_null(), false);
    ///
    /// let x: FeOption<u32> = Unknown;
    /// assert_eq!(x.is_null(), false);
    /// ```
    #[inline]
    pub const fn is_null(&self) -> bool {
        match *self {
            Null => true,
            _ => false,
        }
    }

    /// Returns `true` if the FeOption is an [`Unknown`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ferric::*;
    /// let x: FeOption<u32> = Unknown;
    /// assert_eq!(x.is_unknown(), true);
    ///
    /// let x: FeOption<u32> = Known(2);
    /// assert_eq!(x.is_unknown(), false);
    ///
    /// let x: FeOption<u32> = Null;
    /// assert_eq!(x.is_unknown(), false);
    /// ```
    #[inline]
    pub const fn is_unknown(&self) -> bool {
        match *self {
            Unknown => true,
            _ => false,
        }
    }

    /// Returns `true` if the FeOption is a [`Known`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ferric::*;
    /// let x: FeOption<u32> = Known(2);
    /// assert_eq!(x.is_known(), true);
    ///
    /// let x: FeOption<u32> = Null;
    /// assert_eq!(x.is_known(), false);
    ///
    /// let x: FeOption<u32> = Unknown;
    /// assert_eq!(x.is_known(), false);
    /// ```
    #[inline]
    pub const fn is_known(&self) -> bool {
        match *self {
            Known(_) => true,
            _ => false,
        }
    }

    /// Returns the contained [`Known`] value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the self value equals [`Unknown`] or [`Null`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use ferric::*;
    /// let x = Known("air");
    /// assert_eq!(x.unwrap(), "air");
    /// ```
    ///
    /// ```should_panic
    /// # use ferric::*;
    /// let x: FeOption<&str> = Null;
    /// assert_eq!(x.unwrap(), "air"); // fails
    /// ```
    ///
    /// ```should_panic
    /// # use ferric::*;
    /// let x: FeOption<&str> = Unknown;
    /// assert_eq!(x.unwrap(), "air"); // fails
    /// ```
    pub fn unwrap(self) -> T {
        match self {
            Known(val) => val,
            Null => unwrap_failed_null(),
            Unknown => unwrap_failed_unknown(),
        }
    }
}

// Non-generic, cold panic helpers. Pulling the panic out of the generic
// `unwrap` keeps each panic site at a stable, non-monomorphized source
// location so coverage tools can attribute the line correctly.
#[cold]
#[track_caller]
fn unwrap_failed_null() -> ! {
    panic!("called `FeOption::unwrap()` on a `Null` value")
}

#[cold]
#[track_caller]
fn unwrap_failed_unknown() -> ! {
    panic!("called `FeOption::unwrap()` on an `Unknown` value")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_null_returns_true_only_for_null() {
        let n: FeOption<u32> = Null;
        let u: FeOption<u32> = Unknown;
        let k: FeOption<u32> = Known(7);
        assert!(n.is_null());
        assert!(!u.is_null());
        assert!(!k.is_null());
    }

    #[test]
    fn is_unknown_returns_true_only_for_unknown() {
        let n: FeOption<u32> = Null;
        let u: FeOption<u32> = Unknown;
        let k: FeOption<u32> = Known(7);
        assert!(!n.is_unknown());
        assert!(u.is_unknown());
        assert!(!k.is_unknown());
    }

    #[test]
    fn is_known_returns_true_only_for_known() {
        let n: FeOption<u32> = Null;
        let u: FeOption<u32> = Unknown;
        let k: FeOption<u32> = Known(7);
        assert!(!n.is_known());
        assert!(!u.is_known());
        assert!(k.is_known());
    }

    #[test]
    fn unwrap_known_returns_value() {
        let k: FeOption<&str> = Known("air");
        assert_eq!(k.unwrap(), "air");
    }

    #[test]
    #[should_panic(expected = "called `FeOption::unwrap()` on a `Null` value")]
    fn unwrap_null_panics() {
        let n: FeOption<u32> = Null;
        let _ = n.unwrap();
    }

    #[test]
    #[should_panic(expected = "called `FeOption::unwrap()` on an `Unknown` value")]
    fn unwrap_unknown_panics() {
        let u: FeOption<u32> = Unknown;
        let _ = u.unwrap();
    }

    #[test]
    fn copy_and_clone_are_supported() {
        // Exercise the derived Copy/Clone impls on FeOption for every variant
        // so each arm of the derived `clone` match is hit.
        let k: FeOption<u32> = Known(42);
        let n: FeOption<u32> = Null;
        let u: FeOption<u32> = Unknown;

        // Copy.
        let _k_copy = k;
        let _n_copy = n;
        let _u_copy = u;

        // Clone.
        let k_clone = k.clone();
        let n_clone = n.clone();
        let u_clone = u.clone();

        assert!(k_clone.is_known());
        assert!(n_clone.is_null());
        assert!(u_clone.is_unknown());
    }
}
