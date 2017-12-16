pub mod collections {
    pub mod range {
        use std::collections::Bound;
        use std::ops;

        pub trait RangeArgument<T>
            where T: ?Sized
        {
            fn start(&self) -> Bound<&T>;
            fn end(&self) -> Bound<&T>;
        }

        impl<T> RangeArgument<T> for ops::Range<T> {
            fn start(&self) -> Bound<&T> {
                Bound::Included(&self.start)
            }

            fn end(&self) -> Bound<&T> {
                Bound::Excluded(&self.end)
            }
        }

        impl<T> RangeArgument<T> for ops::RangeFull
            where T: ?Sized
        {
            fn start(&self) -> Bound<&T> {
                Bound::Unbounded
            }

            fn end(&self) -> Bound<&T> {
                Bound::Unbounded
            }
        }

        impl<T> RangeArgument<T> for ops::RangeTo<T> {
            fn start(&self) -> Bound<&T> {
                Bound::Unbounded
            }

            fn end(&self) -> Bound<&T> {
                Bound::Excluded(&self.end)
            }
        }

        impl<T> RangeArgument<T> for ops::RangeFrom<T> {
            fn start(&self) -> Bound<&T> {
                Bound::Included(&self.start)
            }

            fn end(&self) -> Bound<&T> {
                Bound::Unbounded
            }
        }
    }
}

pub mod convert {
    use std::result::Result;

    pub trait TryFrom<T>: Sized {
        type Error;
        fn try_from(T) -> Result<Self, Self::Error>;
    }
}
