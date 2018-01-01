extern crate cuda_sys;

#[macro_use]
mod error;
pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;

pub mod memory;
pub use memory::memcpy;

mod stream;
pub use stream::Stream;

mod misc;
pub use misc::*;

pub mod nightly;
