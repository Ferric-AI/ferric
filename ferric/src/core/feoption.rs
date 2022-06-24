// Copyright 2022 The Ferric AI Project Developers

#[derive(Copy, Clone)]
pub enum FeOption<T> {
    Null,
    Unknown,
    Known(T),
}

impl<T> FeOption<T> {
    #[inline]
    pub const fn is_null(&self) -> bool {
        matches!(*self, FeOption::Null)
    }

    #[inline]
    pub const fn is_unknown(&self) -> bool {
        matches!(*self, FeOption::Unknown)
    }

    #[inline]
    pub const fn is_known(&self) -> bool {
        matches!(*self, FeOption::Known(_))
    }

    #[inline]
    pub fn unwrap(self) -> T {
        match self {
            FeOption::Known(val) => val,
            FeOption::Null => panic!("called `FeOption::unwrap()` on a `Null` value"),
            FeOption::Unknown => panic!("called `FeOption::unwrap()` on an `Unknown` value"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FeOption::{Known, Null, Unknown};

    #[test]
    fn feoption_funcs() {
        let x: FeOption<u32> = Known(2);
        assert_eq!(x.is_null(), false);
        assert_eq!(x.is_unknown(), false);
        assert_eq!(x.is_known(), true);
        assert_eq!(x.unwrap(), 2);
    }

    #[test]
    #[should_panic]
    fn unwrap_null() {
        let x: FeOption<u32> = Null;
        assert_eq!(x.unwrap(), 2);
    }

    #[test]
    #[should_panic]
    fn unwrap_unknown() {
        let x: FeOption<u32> = Unknown;
        assert_eq!(x.unwrap(), 2);
    }
}
