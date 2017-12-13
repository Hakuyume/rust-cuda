use cuda::memory::Repr;

pub fn check_vec<T, X>(n: usize, x: &X, incx: usize)
    where X: Repr<T>
{
    assert!((n - 1) * incx + 1 <= x.len());
}
