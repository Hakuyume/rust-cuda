extern crate cuda;
extern crate cudnn_sys;

#[macro_use]
mod error;
pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;

pub mod context;
pub mod tensor;