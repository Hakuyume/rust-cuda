extern crate cuda_sys;

#[macro_use]
mod error;
pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;

pub mod memory;
pub mod stream;
pub mod misc;
