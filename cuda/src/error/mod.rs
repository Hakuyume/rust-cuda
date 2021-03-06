use std;
use std::ffi;
use std::fmt;

use cuda_sys;

mod error;
pub use self::error::Error;

mod try_from;

impl std::error::Error for Error {
    fn description(&self) -> &str {
        unsafe {
            let ptr = cuda_sys::cudaGetErrorString(*self as _);
            let c_str = ffi::CStr::from_ptr(ptr);
            c_str.to_str().unwrap_or("[Non UTF8 description]")
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

macro_rules! try_call {
    ($call:expr) => {{
        use $crate::nightly::convert::TryFrom;
        try!(match $crate::Error::try_from($call) {
            Ok(err) => Err(err),
            Err(_) => Ok(()),
        })
    }};
}
