mod repr;
pub use self::repr::{Repr, ReprMut};

mod owned_repr;
mod borrowed_repr;

mod array;
pub use self::array::ArrayBase;
pub use self::array::Array;
pub use self::array::{Slice, SliceMut};

mod memcpy;
pub use self::memcpy::memcpy;

#[cfg(test)]
mod tests;
