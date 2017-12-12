extern crate cuda;
extern crate cublas_sys;

#[macro_use]
mod error;
pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;

pub mod scalar;
pub mod context;
pub mod level1;
