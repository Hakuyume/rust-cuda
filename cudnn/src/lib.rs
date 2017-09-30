extern crate cudnn_sys;

pub mod context;
#[macro_use]
mod error;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;
