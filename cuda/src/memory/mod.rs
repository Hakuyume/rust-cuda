mod ptr;
pub use self::ptr::{Ptr, PtrMut};

mod owned;
pub use self::owned::Owned;

mod borrowed;
pub use self::borrowed::{Borrowed, BorrowedMut};

mod array;
pub use self::array::Array;

mod memcpy;
pub use self::memcpy::memcpy;

#[cfg(test)]
mod tests;
