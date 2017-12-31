pub trait Repr {
    type Type;
    fn as_ptr(&self) -> *const Self::Type;
}

pub trait ReprMut: Repr {
    fn as_mut_ptr(&mut self) -> *mut Self::Type;
}
