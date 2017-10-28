mod view;
pub use self::view::{View, ViewMut};

mod memory;
pub use self::memory::Memory;
pub use self::memory::set_malloc_hook;
pub use self::memory::set_free_hook;

mod memcpy;
pub use self::memcpy::memcpy;

#[cfg(test)]
mod tests;
