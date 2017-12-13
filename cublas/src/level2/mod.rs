use std::os::raw::c_int;

use cuda::memory::{Repr, ReprMut};

use Result;
use Operation;
use PointerMode;
use scalar;
use context;
use helper::{check_vec, check_mat};

pub fn gemv<T, A, X, Y>(context: &mut context::Context,
                        trans: Operation,
                        m: usize,
                        n: usize,
                        alpha: &T,
                        a: &A,
                        lda: usize,
                        x: &X,
                        incx: usize,
                        beta: &T,
                        y: &mut Y,
                        incy: usize)
                        -> Result<()>
    where T: scalar::Scalar,
          A: Repr<T>,
          X: Repr<T>,
          Y: ReprMut<T>
{
    assert_eq!(context.get_pointer_mode()?, PointerMode::Host);
    check_mat(m, n, a, lda);
    match trans {
        Operation::N => {
            check_vec(n, x, incx);
            check_vec(m, y, incy);
        }
        _ => {
            check_vec(m, x, incx);
            check_vec(n, y, incy);
        }
    }
    unsafe {
        try_call!(T::GEMV(context.as_mut_ptr(),
                          trans as _,
                          m as c_int,
                          n as c_int,
                          alpha,
                          a.as_ptr(),
                          lda as c_int,
                          x.as_ptr(),
                          incx as c_int,
                          beta,
                          y.as_mut_ptr(),
                          incy as c_int))
    }
    Ok(())
}
