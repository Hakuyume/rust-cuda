use cuda::memory::Repr;

pub fn check_vec<T, X>(n: usize, x: &X, incx: usize)
    where X: Repr<T>
{
    assert!(1 + (n - 1) * incx <= x.len());
}

pub fn check_mat<T, X>(m: usize, n: usize, a: &X, lda: usize)
    where X: Repr<T>
{
    assert!(m <= lda);
    assert!(lda * n <= a.len());
}
