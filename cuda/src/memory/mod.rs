mod view;
pub use self::view::{View, ViewMut};

mod slice;
pub use self::slice::{Slice, SliceMut};
pub use self::slice::{from_raw_parts, from_raw_parts_mut};

mod memory;
pub use self::memory::Memory;
pub use self::memory::{set_malloc_hook, set_free_hook};

mod memcpy;
pub use self::memcpy::memcpy;
