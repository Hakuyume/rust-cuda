use std::os::raw::c_int;

use cuda::memory::{Repr, ReprMut};

use Result;
use scalar;
use context;
use helper::check_vec;

pub fn iamax<T, X>(context: &mut context::Context, n: usize, x: &X, incx: usize) -> Result<usize>
    where T: scalar::Scalar,
          X: Repr<T>
{
    assert_eq!(context.get_pointer_mode()?, context::PointerMode::Host);
    check_vec(n, x, incx);

    let mut result = 0;
    unsafe {
        try_call!(T::IAMAX(context.as_mut_ptr(),
                           n as c_int,
                           x.as_ptr(),
                           incx as c_int,
                           &mut result))
    }
    Ok(result as usize)
}

pub fn axpy<T, X, Y>(context: &mut context::Context,
                     n: usize,
                     alpha: &T,
                     x: &X,
                     incx: usize,
                     y: &mut Y,
                     incy: usize)
                     -> Result<()>
    where T: scalar::Scalar,
          X: Repr<T>,
          Y: ReprMut<T>
{
    assert_eq!(context.get_pointer_mode()?, context::PointerMode::Host);
    check_vec(n, x, incx);
    check_vec(n, y, incy);
    unsafe {
        try_call!(T::AXPY(context.as_mut_ptr(),
                          n as c_int,
                          alpha,
                          x.as_ptr(),
                          incx as c_int,
                          y.as_mut_ptr(),
                          incy as c_int))
    }
    Ok(())
}

pub fn copy<T, X, Y>(context: &mut context::Context,
                     n: usize,
                     x: &X,
                     incx: usize,
                     y: &mut Y,
                     incy: usize)
                     -> Result<()>
    where T: scalar::Scalar,
          X: Repr<T>,
          Y: ReprMut<T>
{
    check_vec(n, x, incx);
    check_vec(n, y, incy);
    unsafe {
        try_call!(T::COPY(context.as_mut_ptr(),
                          n as c_int,
                          x.as_ptr(),
                          incx as c_int,
                          y.as_mut_ptr(),
                          incy as c_int))
    }
    Ok(())
}
