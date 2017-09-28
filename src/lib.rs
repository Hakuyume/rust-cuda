extern crate cuda_sys;

mod error;
pub use error::Error;

pub type Result<T> = std::result::Result<T, Error>;
