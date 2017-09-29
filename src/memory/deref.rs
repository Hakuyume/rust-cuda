use std::ops;
use super::{Memory, Slice};

impl<T> ops::Deref for Memory<T> {
    type Target = Slice<T>;
    fn deref(&self) -> &Slice<T> {
        unsafe { Slice::new(self.ptr, self.len) }
    }
}

impl<T> ops::DerefMut for Memory<T> {
    fn deref_mut(&mut self) -> &mut Slice<T> {
        unsafe { Slice::new_mut(self.ptr, self.len) }
    }
}
