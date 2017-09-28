pub trait Slice<S> {
    type Output;
    fn slice(self, slice: S) -> Self::Output;
}
