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
        matches!(*self, Null)
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
        matches!(*self, Unknown)
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
        matches!(*self, Known(_))
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
    #[inline]
    pub fn unwrap(self) -> T {
        match self {
            Known(val) => val,
            Null => panic!("called `FeOption::unwrap()` on a `Null` value"),
            Unknown => panic!("called `FeOption::unwrap()` on an `Unknown` value"),
        }
    }
}
