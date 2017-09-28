extern crate cuda_sys;

mod error;
pub use error::Error;

mod memory;
pub use memory::Memory;

pub type Result<T> = std::result::Result<T, Error>;
