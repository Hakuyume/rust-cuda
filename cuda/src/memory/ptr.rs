pub trait Ptr {
    type Type;
    fn as_ptr(&self) -> *const Self::Type;
}

pub trait PtrMut: Ptr {
    fn as_mut_ptr(&mut self) -> *mut Self::Type;
}
