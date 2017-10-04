mod memory;
pub use self::memory::Memory;

mod memcpy;
pub use self::memcpy::memcpy;

#[cfg(test)]
mod tests;
