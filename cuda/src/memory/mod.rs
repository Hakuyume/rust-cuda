mod repr;
pub use self::repr::{Repr, ReprMut};

mod view;
pub use self::view::{View, ViewMut};
pub use self::view::{from_raw_parts, from_raw_parts_mut};

mod slice;
pub use self::slice::{Slice, SliceMut};

mod memory;
pub use self::memory::Memory;

mod memcpy;
pub use self::memcpy::memcpy;
